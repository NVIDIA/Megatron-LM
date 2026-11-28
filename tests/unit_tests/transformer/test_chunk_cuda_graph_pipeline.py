# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Chunk-wise CUDA graphs with several model chunks and several live graph slots.

``cuda_graph_impl="chunk_wise"`` captures one forward and one backward graph per model chunk
callable (the decoder block and, with packed sequences, the post-process block of the last
stage) and microbatch slot. The tests build the model chunks of a pipeline stage the way training
does, capture them with ``TECudaGraphHelper`` and replay them through the model's ``forward``,
with pipeline parallelism in the stage's 1F1B order so that several microbatches are in flight
at once, and compare every microbatch with an eager model (``cuda_graph_impl="none"``) holding
the same weights.

All tests need ``torch.distributed.run`` with an even number of ranks: every data-parallel
replica is a pipeline of two stages or a tensor-parallel group of two.
"""

from contextlib import contextmanager

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.packed_seq_params import (
    PackedSeqParams,
    get_thd_padding_kwargs,
    pad_sequence_for_thd,
)
from megatron.core.pipeline_parallel.schedules import (
    get_pp_rank_microbatches,
    get_schedule_table,
    set_current_microbatch,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer import cuda_graphs
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper, convert_schedule_table_to_order
from megatron.core.transformer.moe.moe_logging import (
    MoEMetricsTracker,
    get_moe_metrics_tracker,
    set_moe_metrics_tracker,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_attr_wrapped_model
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.internal,
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        Utils.world_size < 2 or Utils.world_size % 2 != 0,
        reason="requires torch.distributed.run with an even number of ranks",
    ),
]

_HIDDEN = 64
_VOCAB = 128
# Packed (THD) microbatches: token capacity of a pack and bound on its sequences (the padding
# tail is a dummy sequence, so a pack holds at most _MAX_SEQS - 1 documents).
_TOKENS = 64
_MAX_SEQS = 4
_PACKS = ([56], [24, 32], [16, 40], [8, 16, 24])
_PACKED = dict(
    sequence_packing_scheduler="dp_balanced",
    pad_packed_seq_alignment="max",
    max_seqlen_per_dp_cp_rank=_TOKENS,
    thd_max_packed_sequences=_MAX_SEQS,
    cuda_graph_dynamic_microbatches=True,
)
# Error (2-norm, relative to the reference) allowed between a replay and the eager model. A
# replay runs the kernels of the eager forward on the same values; a microbatch that was
# computed from another slot's buffers is off by the order of the reference itself.
_TOLERANCE = 5e-3


def _config(cuda_graph_impl, **settings):
    """Small BF16 model configuration; ``cuda_graph_impl="none"`` gives the eager twin."""
    if cuda_graph_impl == "none":
        settings = {**settings, "cuda_graph_dynamic_microbatches": False}
    return TransformerConfig(
        **{
            "num_layers": 2,
            "hidden_size": _HIDDEN,
            "num_attention_heads": 4,
            "ffn_hidden_size": 2 * _HIDDEN,
            "bf16": True,
            "params_dtype": torch.bfloat16,
            "pipeline_dtype": torch.bfloat16,
            "hidden_dropout": 0.0,
            "attention_dropout": 0.0,
            "use_cpu_initialization": True,
            "use_te_rng_tracker": True,
            "cuda_graph_impl": cuda_graph_impl,
            "cuda_graph_modules": [],
            "cuda_graph_warmup_steps": 1,
            **settings,
        }
    )


def _build_chunks(
    config,
    *,
    position_embedding_type="rope",
    stage_flags=None,
    layer_spec=None,
    hybrid_layer_pattern=None,
):
    """This rank's model chunks (one per virtual pipeline stage), wrapped like training.

    The chunks are ``GPTModel``s, or ``HybridModel``s of ``hybrid_layer_pattern``.
    ``stage_flags`` overrides ``(pre_process, post_process)`` of the single chunk, which lets a
    rank without pipeline parallelism hold the chunk of a later stage.
    """
    if layer_spec is None:
        layer_spec = get_gpt_layer_with_transformer_engine_spec()
    vp_size = config.virtual_pipeline_model_parallel_size
    # Same weights in every run and in both twins: the model-parallel seed covers the weights
    # initialized on the GPU, the global seed those initialized on the CPU (embedding, LM head,
    # MoE routers and experts).
    cpu_rng_state = torch.get_rng_state()
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(42)
    chunks = []
    try:
        for vp_stage in [None] if vp_size is None else range(vp_size):
            pre_process, post_process = stage_flags or (
                parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage),
                parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage),
            )
            if hybrid_layer_pattern is not None:
                model = HybridModel(
                    config=config,
                    hybrid_stack_spec=hybrid_stack_spec,
                    vocab_size=_VOCAB,
                    max_sequence_length=_TOKENS,
                    hybrid_layer_pattern=hybrid_layer_pattern,
                    pre_process=pre_process,
                    post_process=post_process,
                    position_embedding_type=position_embedding_type,
                    vp_stage=vp_stage,
                )
            else:
                mtp_block_spec = None
                if config.mtp_num_layers:
                    mtp_block_spec = get_gpt_mtp_block_spec(
                        config, layer_spec, use_transformer_engine=True, vp_stage=vp_stage
                    )
                model = GPTModel(
                    config=config,
                    transformer_layer_spec=layer_spec,
                    vocab_size=_VOCAB,
                    max_sequence_length=_TOKENS,
                    pre_process=pre_process,
                    post_process=post_process,
                    position_embedding_type=position_embedding_type,
                    mtp_block_spec=mtp_block_spec,
                    vp_stage=vp_stage,
                )
            model = model.cuda()
            model.train()
            chunks.append(
                DistributedDataParallel(
                    config,
                    DistributedDataParallelConfig(),
                    model,
                    disable_bucketing=True,
                    pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
                )
            )
    finally:
        torch.set_rng_state(cpu_rng_state)
    return chunks


def _build_graph_and_eager_chunks(settings, **build_kwargs):
    """Chunk-wise model chunks and their eager twins (same weights, no CUDA graphs)."""
    config = _config("chunk_wise", **settings)
    graph_chunks = _build_chunks(config, **build_kwargs)
    eager_chunks = _build_chunks(_config("none", **settings), **build_kwargs)
    for graph_chunk, eager_chunk in zip(graph_chunks, eager_chunks):
        eager_chunk.module.load_state_dict(graph_chunk.module.state_dict())
    return config, graph_chunks, eager_chunks


def _noise(shape, *seed):
    """Deterministic BF16 values, different for every seed tuple."""
    value = 0
    for item in seed:
        value = value * 131 + int(item) + 1
    generator = torch.Generator(device="cpu").manual_seed(value)
    return torch.randn(shape, generator=generator).to(device="cuda", dtype=torch.bfloat16)


def _packed_microbatch(seqlens, seed):
    """One packed microbatch padded to the static THD capacity like the data pipeline pads it
    (``--pad-packed-seq-alignment max --thd-max-packed-sequences N``: tokens to the capacity
    with a dummy tail sequence, cu_seqlens to ``N + 1`` entries); identical on every rank."""
    total = sum(seqlens)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randint(0, _VOCAB, (1, total + 1), generator=generator).cuda()
    cu_seqlens = torch.tensor([0, *seqlens], dtype=torch.int32, device="cuda").cumsum(0).int()
    alignment, target_len, max_num_seqs = get_thd_padding_kwargs(
        pad_packed_seq_alignment="max",
        max_seqlen_per_dp_cp_rank=_TOKENS,
        thd_max_packed_sequences=_MAX_SEQS,
        cuda_graph_static=True,
    )
    input_ids, labels, loss_mask, position_ids, packed_seq_params, padding_mask = (
        pad_sequence_for_thd(
            tokens[:, :-1].contiguous(),
            tokens[:, 1:].contiguous(),
            torch.ones(1, total, dtype=torch.float32, device="cuda"),
            torch.cat([torch.arange(length) for length in seqlens]).unsqueeze(0).cuda(),
            PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens.clone(),
                cu_seqlens_q_padded=cu_seqlens.clone(),
                cu_seqlens_kv_padded=cu_seqlens.clone(),
                max_seqlen_q=max(seqlens),
                max_seqlen_kv=max(seqlens),
            ),
            alignment=alignment,
            target_len=target_len,
            max_num_seqs=max_num_seqs,
        )
    )
    assert input_ids.shape == (1, _TOKENS)
    assert packed_seq_params.cu_seqlens_q.numel() == _MAX_SEQS + 1
    return dict(
        input_ids=input_ids,
        position_ids=position_ids,
        labels=labels,
        loss_mask=loss_mask,
        padding_mask=padding_mask,
        packed_seq_params=packed_seq_params,
    )


def _packed_feed(model, batch, hidden_shape, *seed):
    """What one model chunk consumes for one packed microbatch.

    ``inputs`` is what the packing scheduler hands the stage (tokens only where they are
    embedded, labels only where the loss is computed), ``hidden`` what a later stage receives
    from its predecessor and ``output_grad`` what an earlier stage receives from its successor.
    """
    core = model.module
    owns_tokens = core.pre_process or core.mtp_process
    owns_labels = core.post_process or core.mtp_process
    return dict(
        inputs=dict(
            input_ids=batch["input_ids"] if owns_tokens else None,
            position_ids=batch["position_ids"] if owns_tokens else None,
            attention_mask=None,
            labels=batch["labels"] if owns_labels else None,
            loss_mask=batch["loss_mask"] if owns_labels else None,
            packed_seq_params=batch["packed_seq_params"],
            padding_mask=batch["padding_mask"],
        ),
        hidden=None if core.pre_process else _noise(hidden_shape, 0, *seed),
        output_grad=None if core.post_process else _noise(hidden_shape, 1, *seed),
        loss_mask=batch["loss_mask"] if core.post_process else None,
    )


def _forward(model, feed, microbatch=0):
    """Forward of one model chunk as ``schedules.forward_step`` runs it."""
    set_current_microbatch(model, microbatch)
    hidden = feed["hidden"]
    if hidden is not None:
        # The tensor received from the previous stage is a new leaf for every microbatch.
        hidden = hidden.clone().requires_grad_(True)
    get_attr_wrapped_model(model, "set_input_tensor")([hidden])
    return hidden, model(**feed["inputs"])


def _backward(model, hidden, output, feed):
    """Backward of one model chunk as ``schedules.backward_step`` runs it.

    Returns the gradient of the pipeline input and this microbatch's parameter gradients."""
    model.zero_grad_buffer()
    if feed["output_grad"] is not None:
        torch.autograd.backward(output, grad_tensors=feed["output_grad"])
    else:
        loss_mask = feed["loss_mask"]
        (torch.sum(output.float() * loss_mask) / loss_mask.sum()).backward()
    return (
        None if hidden is None else hidden.grad.detach().float().clone(),
        {
            name: param.main_grad.detach().float().clone()
            for name, param in model.module.named_parameters()
            if param.requires_grad
        },
    )


def _step(model, feed):
    """Forward and backward of one microbatch without pipelining."""
    hidden, output = _forward(model, feed)
    # A replayed output is a view of a static graph buffer, valid until its backward.
    result = {"output": output.detach().float().clone()}
    result["input_grad"], result["grads"] = _backward(model, hidden, output, feed)
    return result


def _pipeline_order(config, num_microbatches):
    """This rank's forward / backward order (1-based chunk ids, negative for backward)."""
    num_chunks = config.virtual_pipeline_model_parallel_size or 1
    group_size = config.microbatch_group_size_per_vp_stage
    _, _, num_warmup_microbatches, _ = get_pp_rank_microbatches(
        num_microbatches, num_chunks, group_size, forward_only=False
    )
    return convert_schedule_table_to_order(
        num_warmup_microbatches,
        num_chunks,
        get_schedule_table(num_microbatches, num_chunks, group_size),
    )


def _run_pipeline_order(models, feeds, order, after_forward=None):
    """Drive this rank's model chunks through ``order``, several microbatches in flight.

    Returns ``{(chunk, microbatch): result}`` and the largest number of microbatches one chunk
    held between its forward and its backward."""
    forwards, backwards = [0] * len(models), [0] * len(models)
    in_flight, results, peak = {}, {}, 0
    for entry in order:
        chunk = abs(entry) - 1
        if entry > 0:
            microbatch = forwards[chunk]
            forwards[chunk] += 1
            hidden, output = _forward(models[chunk], feeds[chunk][microbatch], microbatch)
            if after_forward is not None:
                after_forward(chunk, microbatch)
            results[chunk, microbatch] = {"output": output.detach().float().clone()}
            in_flight[chunk, microbatch] = hidden, output
            peak = max(peak, sum(key[0] == chunk for key in in_flight))
        else:
            microbatch = backwards[chunk]
            backwards[chunk] += 1
            hidden, output = in_flight.pop((chunk, microbatch))
            result = results[chunk, microbatch]
            result["input_grad"], result["grads"] = _backward(
                models[chunk], hidden, output, feeds[chunk][microbatch]
            )
    assert not in_flight
    return results, peak


def _relative_error(actual, expected):
    """2-norm of the difference relative to the reference (absolute for a zero reference)."""
    error, scale = (actual - expected).norm().item(), expected.norm().item()
    return error / scale if scale > 0 else error


def _assert_result_matches(result, expected, what):
    """Output, input gradient and every parameter gradient of one microbatch."""

    def check(actual, reference, part):
        assert actual.shape == reference.shape, (what, part, actual.shape, reference.shape)
        assert torch.isfinite(actual).all(), (what, part)
        error = _relative_error(actual, reference)
        assert error <= _TOLERANCE, f"{what}, {part}: relative error {error:.3e}"

    check(result["output"], expected["output"], "output")
    assert (result["input_grad"] is None) == (expected["input_grad"] is None), what
    if expected["input_grad"] is not None:
        check(result["input_grad"], expected["input_grad"], "input gradient")
    assert result["grads"].keys() == expected["grads"].keys(), what
    for name, grad in expected["grads"].items():
        check(result["grads"][name], grad, f"gradient of {name}")


def _largest_gradient_change(result, expected):
    return max(
        _relative_error(result["grads"][name], grad) for name, grad in expected["grads"].items()
    )


@contextmanager
def _training_stream():
    """A non-default CUDA stream for building, warming up and replaying. Training wraps its
    model chunks on a dedicated stream, so that the AccumulateGrad nodes the DDP wrapper keeps
    alive do not reference the default stream."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        yield
    torch.cuda.current_stream().wait_stream(stream)


@contextmanager
def _captured_graphs(models, config, seq_length, micro_batch_size):
    """Capture the chunk graphs of ``models``; delete them afterwards, also on failure."""
    helper = TECudaGraphHelper(models, config, seq_length, micro_batch_size, optimizers=[])
    try:
        helper.create_cudagraphs()
        assert helper.graphs_created()
        yield helper
    finally:
        torch.cuda.synchronize()
        if helper.graphs_created():
            helper.delete_cuda_graphs()
        elif cuda_graphs.is_graph_capturing():
            # A capture that raised can leave the process-wide capture flag set.
            cuda_graphs._set_capture_end()


@contextmanager
def _recorded_slots(helper):
    """Record ``(callable class, slot)`` for every graph a block callable replays."""
    log, originals = [], []

    def recording(graph, key):
        def replay(*args, **kwargs):
            log.append(key)
            return graph(*args, **kwargs)

        return replay

    for block in helper.flattened_callables:
        originals.append((block, block.te_cuda_graphs))
        block.te_cuda_graphs = [
            recording(graph, (type(block).__name__, slot))
            for slot, graph in enumerate(block.te_cuda_graphs)
        ]
    try:
        yield log
    finally:
        for block, graphs in originals:
            block.te_cuda_graphs = graphs


class TestChunkCudaGraphPipeline:

    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        # The process-wide MoE metrics tracker keeps its entries, sized by the layer count of
        # the first model that recorded into it: every test records into a tracker of its own.
        self._moe_metrics_tracker = get_moe_metrics_tracker()
        set_moe_metrics_tracker(MoEMetricsTracker())

    def teardown_method(self, method):
        set_moe_metrics_tracker(self._moe_metrics_tracker)
        destroy_num_microbatches_calculator()
        Utils.destroy_model_parallel()

    @staticmethod
    def _init_num_microbatches(num_microbatches, micro_batch_size=1):
        # The capture sizes its slots and its Transformer Engine warm-up from this count.
        init_num_microbatches_calculator(
            rank=0,
            rampup_batch_size=None,
            global_batch_size=num_microbatches * micro_batch_size,
            micro_batch_size=micro_batch_size,
            data_parallel_size=1,
        )

    @pytest.mark.parametrize("vp_size,mtp_layers", [(None, 0), (2, 1)])
    def test_packed_pipeline_replay_matches_eager_per_microbatch(self, vp_size, mtp_layers):
        """Two pipeline stages (with two virtual stages each) of packed sequences.

        A stage holds one model chunk per virtual stage; the last chunk of the last stage also
        owns the post-process callable (with the MTP block when the model has one). Every chunk
        is driven through ``GPTModel.forward`` in the stage's 1F1B order with a different pack
        per microbatch: later chunks take their hidden states from ``set_input_tensor``, every
        callable replays the slot of its microbatch, and a microbatch beyond the captured slots
        is refused.
        """
        Utils.initialize_model_parallel(
            pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=vp_size
        )
        num_chunks, num_microbatches = vp_size or 1, len(_PACKS)
        self._init_num_microbatches(num_microbatches)
        settings = dict(
            num_layers=2 * 2 * num_chunks,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=vp_size,
            microbatch_group_size_per_vp_stage=2,
            pipeline_p2p_fixed_shape=True,
            mtp_num_layers=mtp_layers or None,
            **_PACKED,
        )
        with _training_stream():
            config, models, eager_models = _build_graph_and_eager_chunks(settings)
            first_stage = parallel_state.is_pipeline_first_stage()
            last_stage = parallel_state.is_pipeline_last_stage()
            cores = [model.module for model in models]
            assert [core.pre_process for core in cores] == [first_stage] + [False] * (
                num_chunks - 1
            )
            assert [core.post_process for core in cores] == [False] * (num_chunks - 1) + [
                last_stage
            ]
            assert (cores[-1].postprocess_block is not None) == last_stage
            assert cores[-1].mtp_process == (last_stage and bool(mtp_layers))

            batches = [_packed_microbatch(seqlens, seed) for seed, seqlens in enumerate(_PACKS)]
            feeds = [
                [
                    _packed_feed(model, batch, (_TOKENS, 1, _HIDDEN), Utils.rank, chunk, microbatch)
                    for microbatch, batch in enumerate(batches)
                ]
                for chunk, model in enumerate(models)
            ]
            order = _pipeline_order(config, num_microbatches)

            reference, peak = _run_pipeline_order(eager_models, feeds, order)
            if first_stage or num_chunks > 1:
                assert peak >= 2, "a chunk of this stage must hold several microbatches in flight"
            for chunk in range(num_chunks):
                for microbatch in range(1, num_microbatches):
                    assert (
                        _relative_error(
                            reference[chunk, microbatch]["output"],
                            reference[chunk, microbatch - 1]["output"],
                        )
                        > _TOLERANCE
                    ), "microbatches must differ"
            # The eager step before the capture (cuda_graph_warmup_steps): the chunk-wise
            # model's own forward, which already goes through the post-process block.
            warmup, _ = _run_pipeline_order(models, feeds, order)
            for key, result in warmup.items():
                _assert_result_matches(result, reference[key], f"eager warm-up {key}")

            with _captured_graphs(models, config, _TOKENS, 1) as helper:
                # One slot per microbatch for every callable of every chunk.
                assert helper.num_microbatches == num_microbatches
                assert helper.num_layers_per_chunk == [1] * (num_chunks - 1) + [
                    2 if last_stage else 1
                ]
                for callables in helper.callables_per_chunk:
                    for block in callables:
                        assert len(block.te_cuda_graphs) == num_microbatches

                with _recorded_slots(helper) as replayed:

                    def check_slots(chunk, microbatch):
                        expected = [
                            (type(block).__name__, microbatch)
                            for block in helper.callables_per_chunk[chunk]
                        ]
                        assert replayed == expected, (chunk, microbatch)
                        replayed.clear()

                    # A packed step can hold fewer microbatches than the capture has slots.
                    for count in (num_microbatches, 2, num_microbatches):
                        results, _ = _run_pipeline_order(
                            models, feeds, _pipeline_order(config, count), check_slots
                        )
                        assert len(results) == num_chunks * count
                        for key, result in results.items():
                            _assert_result_matches(
                                result, reference[key], f"replay of {count} microbatches {key}"
                            )

                # The capture has no slot for one more microbatch in flight.
                with pytest.raises(
                    RuntimeError, match=f"only contains {num_microbatches} graph slots"
                ):
                    _forward(models[0], feeds[0][0], microbatch=num_microbatches)

    def test_fixed_length_mrope_slots_keep_their_own_position_table(self):
        """Fixed-length mRoPE on two pipeline stages, position ids differing per microbatch.

        The rotary table is a graph input built from the position ids, and the fused rotation
        reads it again in its backward. The first stage runs forward(0), forward(1),
        backward(0), ...: microbatch 0 is differentiated with its own table although
        microbatch 1 was forwarded in between.
        """
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        num_microbatches, micro_batch_size, seq_length = 4, 2, 32
        self._init_num_microbatches(num_microbatches, micro_batch_size)
        settings = dict(
            num_layers=4,
            pipeline_model_parallel_size=2,
            microbatch_group_size_per_vp_stage=2,
            # kv_channels 16: the T / H / W sections span its 8 rotary frequencies.
            mrope_section=[4, 2, 2],
            apply_rope_fusion=True,
            # Attention scores of order one: the path through the rotated queries and keys then
            # carries a visible part of every gradient (at the default initialization the
            # scores, and with them that path, are close to zero).
            init_method_std=0.1,
            # The batches carry no attention mask (causal attention).
            create_attention_mask_in_dataloader=False,
        )
        with _training_stream():
            config, models, eager_models = _build_graph_and_eager_chunks(
                settings, position_embedding_type="mrope"
            )
            core = models[0].module
            hidden_shape = (seq_length, micro_batch_size, _HIDDEN)
            feeds = []
            for microbatch in range(num_microbatches):
                generator = torch.Generator(device="cpu").manual_seed(microbatch)
                tokens = torch.randint(
                    0, _VOCAB, (micro_batch_size, seq_length + 1), generator=generator
                ).cuda()
                # [3, batch, seq] temporal / height / width positions of a multimodal sample.
                position_ids = torch.randint(
                    0, seq_length, (3, micro_batch_size, seq_length), generator=generator
                ).cuda()
                feeds.append(
                    dict(
                        inputs=dict(
                            input_ids=tokens[:, :-1].contiguous(),
                            position_ids=position_ids,
                            attention_mask=None,
                            labels=tokens[:, 1:].contiguous() if core.post_process else None,
                        ),
                        hidden=(
                            None
                            if core.pre_process
                            else _noise(hidden_shape, 0, Utils.rank, microbatch)
                        ),
                        output_grad=(
                            None
                            if core.post_process
                            else _noise(hidden_shape, 1, Utils.rank, microbatch)
                        ),
                        loss_mask=torch.ones(micro_batch_size, seq_length, device="cuda"),
                    )
                )
            feeds = [feeds]
            order = _pipeline_order(config, num_microbatches)
            reference, peak = _run_pipeline_order(eager_models, feeds, order)
            if core.pre_process:
                assert order[:3] == [1, 1, -1] and peak == 2

            with _captured_graphs(models, config, seq_length, micro_batch_size):
                # Fixed-length batches keep the eager post-process.
                assert core.postprocess_block is None
                assert len(core.decoder.te_cuda_graphs) == num_microbatches
                for step in range(2):
                    results, _ = _run_pipeline_order(models, feeds, order)
                    for key, result in results.items():
                        _assert_result_matches(result, reference[key], f"step {step} replay {key}")

    @pytest.mark.parametrize("attention", ["transformer_engine", "megatron"])
    @pytest.mark.parametrize("attention_mask", ["from_dataloader", "none"])
    def test_fixed_length_graph_captured_with_the_dataloader_attention_mask(
        self, attention_mask, attention
    ):
        """Fixed-length batches with ``create_attention_mask_in_dataloader`` (the default).

        The block graph takes the dataloader's attention mask as a static input. The training
        script hands that mask to the first and the last pipeline stage only: a stage in between
        gets no batch at all and calls its block with ``attention_mask=None``. Both calls must
        replay, and match the eager model called the same way. Transformer Engine's attention
        applies its causal mask itself; Megatron's own attention (``fallback_to_eager_attn``)
        applies a given mask as its only mask, so a replay that passed a mask without the causal
        part there would let every token attend to its successors."""
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        num_microbatches, micro_batch_size, seq_length = 4, 2, 32
        self._init_num_microbatches(num_microbatches, micro_batch_size)
        settings = dict(
            num_layers=4, pipeline_model_parallel_size=2, microbatch_group_size_per_vp_stage=2
        )
        build_kwargs = {}
        if attention == "megatron":
            settings["fallback_to_eager_attn"] = True
            build_kwargs["layer_spec"] = get_gpt_layer_with_transformer_engine_spec(
                fallback_to_eager_attn=True
            )
        with _training_stream():
            config, models, eager_models = _build_graph_and_eager_chunks(settings, **build_kwargs)
            assert config.create_attention_mask_in_dataloader
            core = models[0].module
            hidden_shape = (seq_length, micro_batch_size, _HIDDEN)
            mask = None
            if attention_mask == "from_dataloader":
                # get_ltor_masks_and_position_ids: True above the diagonal.
                mask = ~torch.tril(
                    torch.ones((micro_batch_size, 1, seq_length, seq_length), device="cuda")
                ).bool()
            feeds = []
            for microbatch in range(num_microbatches):
                generator = torch.Generator(device="cpu").manual_seed(microbatch)
                tokens = torch.randint(
                    0, _VOCAB, (micro_batch_size, seq_length + 1), generator=generator
                ).cuda()
                feeds.append(
                    dict(
                        inputs=dict(
                            input_ids=tokens[:, :-1].contiguous(),
                            position_ids=torch.arange(seq_length, device="cuda").expand(
                                micro_batch_size, -1
                            ),
                            attention_mask=mask,
                            labels=tokens[:, 1:].contiguous() if core.post_process else None,
                        ),
                        hidden=(
                            None
                            if core.pre_process
                            else _noise(hidden_shape, 0, Utils.rank, microbatch)
                        ),
                        output_grad=(
                            None
                            if core.post_process
                            else _noise(hidden_shape, 1, Utils.rank, microbatch)
                        ),
                        loss_mask=torch.ones(micro_batch_size, seq_length, device="cuda"),
                    )
                )
            feeds = [feeds]
            order = _pipeline_order(config, num_microbatches)
            reference, _ = _run_pipeline_order(eager_models, feeds, order)

            with _captured_graphs(models, config, seq_length, micro_batch_size):
                assert len(core.decoder.te_cuda_graphs) == num_microbatches
                for step in range(2):
                    results, _ = _run_pipeline_order(models, feeds, order)
                    for key, result in results.items():
                        _assert_result_matches(result, reference[key], f"step {step} replay {key}")

    def test_deferred_lm_head_weight_gradient_uses_every_microbatch(self):
        """``defer_embedding_wgrad_compute`` on the last pipeline stage, fixed-length batches.

        The eager LM head keeps its input of every microbatch, and the schedule computes the
        weight gradient from them after the last backward of the step. That input is the output
        of the replayed decoder block: a static buffer of the block's graph, which Transformer
        Engine lends to later graphs once the block's backward has run. Every kept input must
        still be its own microbatch's when the schedule drains them.
        """
        pytest.importorskip("fused_weight_gradient_mlp_cuda")
        from megatron.core.pipeline_parallel.schedules import finish_embedding_wgrad_compute

        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        num_microbatches, micro_batch_size, seq_length = 4, 2, 32
        self._init_num_microbatches(num_microbatches, micro_batch_size)
        settings = dict(
            num_layers=4,
            pipeline_model_parallel_size=2,
            microbatch_group_size_per_vp_stage=2,
            defer_embedding_wgrad_compute=True,
            gradient_accumulation_fusion=True,
            create_attention_mask_in_dataloader=False,
        )
        with _training_stream():
            # Every rank holds the chunk of the last stage: decoder layers and the LM head.
            config, models, eager_models = _build_graph_and_eager_chunks(
                settings, stage_flags=(False, True)
            )
            hidden_shape = (seq_length, micro_batch_size, _HIDDEN)
            feeds = []
            for microbatch in range(num_microbatches):
                generator = torch.Generator(device="cpu").manual_seed(microbatch)
                tokens = torch.randint(
                    0, _VOCAB, (micro_batch_size, seq_length + 1), generator=generator
                ).cuda()
                feeds.append(
                    dict(
                        inputs=dict(
                            input_ids=tokens[:, :-1].contiguous(),
                            position_ids=torch.arange(seq_length, device="cuda").expand(
                                micro_batch_size, -1
                            ),
                            attention_mask=None,
                            labels=tokens[:, 1:].contiguous(),
                        ),
                        hidden=_noise(hidden_shape, 0, Utils.rank, microbatch),
                        output_grad=None,
                        loss_mask=torch.ones(micro_batch_size, seq_length, device="cuda"),
                    )
                )
            feeds = [feeds]
            order = _pipeline_order(config, num_microbatches)

            def step_with_deferred_weight_gradient(chunks):
                """One step of the schedule: the microbatches, then the deferred GEMMs."""
                core = chunks[0].module
                core.embedding_activation_buffer.clear()
                core.grad_output_buffer.clear()
                results, _ = _run_pipeline_order(chunks, feeds, order)
                assert len(core.embedding_activation_buffer) == num_microbatches
                chunks[0].zero_grad_buffer()
                finish_embedding_wgrad_compute(
                    config, core, True, parallel_state.get_tensor_model_parallel_group()
                )
                assert not core.embedding_activation_buffer and not core.grad_output_buffer
                return results, core.output_layer.weight.main_grad.detach().float().clone()

            reference, reference_weight_grad = step_with_deferred_weight_gradient(eager_models)
            assert reference_weight_grad.norm() > 0
            with _captured_graphs(models, config, seq_length, micro_batch_size):
                assert len(models[0].module.decoder.te_cuda_graphs) == num_microbatches
                for step in range(2):
                    results, weight_grad = step_with_deferred_weight_gradient(models)
                    for key, result in results.items():
                        _assert_result_matches(result, reference[key], f"step {step} replay {key}")
                    error = _relative_error(weight_grad, reference_weight_grad)
                    assert (
                        error <= _TOLERANCE
                    ), f"step {step}: the deferred LM head weight gradient is off by {error:.3e}"

    @pytest.mark.parametrize("metadata", ["full_length", "sp_sharded", "hybrid"])
    def test_later_stage_token_metadata_under_sequence_parallelism(self, metadata):
        """Packed sequences, TP 2 with sequence parallelism, a chunk without the embedding.

        Such a block sees half of the pack's tokens, and its graph takes ``padding_mask`` and
        the hash router's ``input_ids`` in that frame, whatever the model hands the block.
        ``full_length``: a ``GPTModel`` passes a later stage the full-length metadata.
        ``hybrid``: a ``HybridModel`` scatters the mask of a later stage to the block's tokens
        (its hash routers live on the stage that embeds, which gets the ids scattered too).
        ``sp_sharded``: a ``GPTModel`` block that is handed the shard of both, mask and ids.
        The MoE layers route by learned top-k with a load-balancing loss over the tokens the
        padding mask leaves, the first one of a ``GPTModel`` by token id, so the gradients
        depend on the metadata each rank's graph received.
        """
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, expert_tensor_parallel_size=1)
        self._init_num_microbatches(1)
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        local_tokens = _TOKENS // 2
        hybrid = metadata == "hybrid"
        settings = dict(
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            expert_tensor_parallel_size=1,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_ffn_hidden_size=2 * _HIDDEN,
            # Static token shapes for the captured dispatch: every expert is padded to its
            # capacity, which holds all of the rank's tokens.
            moe_token_dispatcher_type="alltoall",
            moe_expert_capacity_factor=2.0,
            moe_pad_expert_input_to_capacity=True,
            # A load-balancing loss weighted to be a visible part of the router gradients.
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=10.0,
            **_PACKED,
        )
        if hybrid:
            # Attention and MoE are separate layers of the HybridStack.
            settings.update(num_layers=4, is_hybrid_model=True)
            build_kwargs = dict(hybrid_layer_pattern="*E*E")
        else:
            settings.update(moe_num_hash_layers=1, hash_moe_vocab_size=_VOCAB)
            build_kwargs = dict(
                layer_spec=get_gpt_layer_with_transformer_engine_spec(
                    num_experts=4, moe_grouped_gemm=False
                )
            )
        with _training_stream():
            config, (model,), (eager_model,) = _build_graph_and_eager_chunks(
                settings, stage_flags=(False, False), **build_kwargs
            )

            def shard(value):
                if metadata == "full_length":
                    return value
                return value[:, tp_rank * local_tokens : (tp_rank + 1) * local_tokens].contiguous()

            feeds = []
            for seed, seqlens in enumerate(([40], [12, 24], [20, 8, 16])):
                batch = _packed_microbatch(seqlens, seed)
                # The two tensor-parallel ranks own different parts of the mask.
                assert not batch["padding_mask"][:, :local_tokens].any()
                assert 0 < batch["padding_mask"][:, local_tokens:].sum() < local_tokens
                # (The feed of a chunk that neither embeds nor predicts has the full-length
                # mask and no token ids.)
                feed = _packed_feed(model, batch, (local_tokens, 1, _HIDDEN), Utils.rank, seed)
                if not hybrid:
                    feed["inputs"].update(
                        padding_mask=shard(batch["padding_mask"]),
                        input_ids=shard(batch["input_ids"]),
                    )
                feeds.append(feed)
            reference = [_step(eager_model, feed) for feed in feeds]

            # The comparison is only as strong as the routers' use of the metadata: an eager
            # step without padding, or with other token ids, gives different gradients.
            inputs = feeds[0]["inputs"]
            others = {"padding_mask": torch.zeros_like(inputs["padding_mask"])}
            if not hybrid:
                others["input_ids"] = (inputs["input_ids"] + 1) % _VOCAB
            for key, other in others.items():
                changed = _step(eager_model, dict(feeds[0], inputs={**inputs, key: other}))
                assert _largest_gradient_change(changed, reference[0]) > 10 * _TOLERANCE, key

            with _captured_graphs([model], config, _TOKENS, 1) as helper:
                assert helper.flattened_callables == [model.module.decoder]
                for order in ((0, 1, 2), (2, 0, 1)):
                    for index in order:
                        _assert_result_matches(
                            _step(model, feeds[index]), reference[index], f"replay of pack {index}"
                        )

    def test_full_recompute_with_distributed_saved_activations(self):
        """Full recompute, ``distribute_saved_activations``, TP 2 without sequence parallelism.

        A distributing checkpoint shards its input in place; for the first checkpoint of a block
        that input is the graph's static input buffer. Every replay computes on the microbatch
        it was given.
        """
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        self._init_num_microbatches(1)
        settings = dict(
            tensor_model_parallel_size=2,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
            distribute_saved_activations=True,
            **_PACKED,
        )
        with _training_stream():
            config, (model,), (eager_model,) = _build_graph_and_eager_chunks(settings)
            feeds = [
                _packed_feed(model, _packed_microbatch(seqlens, seed), None, seed)
                for seed, seqlens in enumerate(_PACKS)
            ]
            reference = [_step(eager_model, feed) for feed in feeds]
            for first, second in zip(reference, reference[1:]):
                assert (
                    _relative_error(first["output"], second["output"]) > _TOLERANCE
                ), "microbatches must differ"

            with _captured_graphs([model], config, _TOKENS, 1) as helper:
                # Decoder block and post-process block.
                assert len(helper.flattened_callables) == 2
                for order in ((0, 1, 2, 3), (3, 1, 0, 2)):
                    for index in order:
                        _assert_result_matches(
                            _step(model, feeds[index]), reference[index], f"replay of pack {index}"
                        )
