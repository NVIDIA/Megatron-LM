# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Recurrent-state and real-inner-MTP complements to the dense BI campaign."""

from functools import partial

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.inference.engines import batch_invariant_test_utils as bi


def _build_model_engine(case, backend, version, *, mamba, engines, patch):
    torch.manual_seed(321)
    model_parallel_cuda_manual_seed(321, inference_rng_tracker=True, force_reset_rng=True)
    depth = case.context.get("num_speculative_tokens", 0)
    graph = case.context.get("num_cuda_graphs") is not None
    cfg = TransformerConfig(
        **bi.model_defaults(backend, version),
        num_layers=3 if mamba else 2,
        hidden_size=256 if mamba else 128,
        num_attention_heads=8 if mamba else 4,
        mamba_num_heads=16,
        mamba_head_dim=32,
        mamba_num_groups=8,
        mamba_state_dim=16,
        is_hybrid_model=mamba,
        mtp_num_layers=depth or None,
        tensor_model_parallel_size=case.tp,
        pipeline_model_parallel_size=case.pp,
        sequence_parallel=case.sp,
        cuda_graph_impl="local" if graph else "none",
        inference_cuda_graph_scope=(
            InferenceCudaGraphScope.block if graph else InferenceCudaGraphScope.none
        ),
    )
    options = dict(
        config=cfg,
        vocab_size=bi.VOCAB,
        max_sequence_length=1024,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
    )
    if mamba:
        model = HybridModel(
            **options, hybrid_stack_spec=hybrid_stack_spec, hybrid_layer_pattern="M*-"
        )
    else:
        spec = get_gpt_layer_with_transformer_engine_spec()
        model = GPTModel(
            **options,
            transformer_layer_spec=spec,
            mtp_block_spec=get_gpt_mtp_block_spec(cfg, spec, use_transformer_engine=True),
        )
    model = model.cuda().eval()
    assert model.embedding.word_embeddings.weight.dtype == torch.bfloat16
    assert model.output_layer.weight.dtype == torch.bfloat16
    state_config = MambaInferenceStateConfig.from_model(model)
    context_options = dict(
        max_sequence_length=1024,
        max_requests=128 if mamba else 68,
        max_tokens=512,
        block_size_tokens=256,
        buffer_size_gb=0.25,
        mamba_inference_state_config=state_config,
        mamba_memory_ratio=0.5 if mamba else None,
        materialize_only_last_token_logits=False,
        use_cuda_graphs_for_non_decode_steps=False,
        async_sched_mode="legacy",
    )
    context_options.update(case.context)
    ctx = DynamicInferenceContext(cfg, InferenceConfig(**context_options))
    if mamba:
        assert ctx.mamba_ssm_states.dtype == torch.float32
    controller = TextGenerationController(
        GPTInferenceWrapper(Float16Module(cfg, model).eval(), ctx), bi.DummyTokenizer(bi.VOCAB)
    )
    evidence = dict(states=[], mtp=[], inner_calls=set(), replayed_depths=set())

    def target_index():
        selected = (ctx.request_ids[: ctx.total_request_count] == bi.TARGET).nonzero().flatten()
        return int(selected[0]) if selected.numel() else None

    if depth:
        for d, layer in enumerate(model.mtp.layers):

            def inner_hook(module, args, kwargs, *, d=d):
                if torch.cuda.is_current_stream_capturing():
                    ctx._bi_capturing_graph.__dict__.setdefault("_bi_mtp_depths", set()).add(d)
                elif target_index() is not None:
                    evidence["inner_calls"].add(d)

            layer.mtp_model_layer.register_forward_pre_hook(inner_hook, with_kwargs=True)

    engine = DynamicInferenceEngine(controller, ctx)
    engine._bi_model_evidence = evidence
    engines.append(engine)
    if depth:
        original_mtp = model.compute_mtp_single_step

        def mtp_step(*args, **kwargs):
            idx = target_index()
            row = idx - ctx.paused_request_count if idx is not None else None
            if row is not None:
                request = engine.get_request(bi.TARGET)
                key = (
                    int(kwargs["depth"]),
                    int(kwargs["position_ids"][0, row]),
                    int(kwargs["next_token_ids"][0, row]),
                    tuple(request.generated_tokens),
                )
                requests = ctx.total_request_count - ctx.paused_request_count
                graph_step = ctx.using_cuda_graph_this_step()
            result = original_mtp(*args, **kwargs)
            if row is not None:
                logits = result[1]
                expected_rows = ctx.padded_batch_dimensions.req_count if graph_step else requests
                assert logits.shape == (expected_rows, 1, bi.VOCAB)
                evidence["mtp"].append(
                    (key, requests, tuple(logits.shape), logits[row, 0].detach().clone().cpu())
                )
            return result

        patch.setattr(model, "compute_mtp_single_step", mtp_step)

    def record_mamba_state():
        idx = target_index()
        # Capture identity before bookkeeping can recycle the target's recurrent slot.
        end = (
            int(ctx.request_kv_length_offsets[idx] + ctx.request_query_lengths[idx])
            if idx is not None
            else None
        )
        if mamba and idx is not None:
            slot = int(ctx.mamba_metadata.request_to_mamba_state_idx[idx])
            counts = []
            for mixer in model.modules():
                if isinstance(mixer, MambaMixer):
                    decoder = mixer._batch_invariant_decoder
                    count = int(decoder.buffers.num_buffered[slot])
                    assert count == end % mixer.chunk_size, (count, end, mixer.chunk_size)
                    counts.append(count)
            assert counts, "target never reached a buffered Mamba decoder"
            evidence["states"].append(
                (
                    end,
                    ctx.is_decode_only(),
                    slot,
                    counts,
                    ctx.mamba_ssm_states[:, slot].detach().clone(),
                )
            )

    engine._bi_after_forward = record_mamba_state
    if depth and graph:
        manager = model._mtp_cudagraph_manager
        for runner in manager.cudagraph_runners:
            original_replay = runner.fwd_graph.replay

            def replay(*, original_replay=original_replay, graph=runner.fwd_graph):
                result = original_replay()
                if target_index() is not None:
                    evidence["replayed_depths"].update(graph.__dict__.get("_bi_mtp_depths", ()))
                return result

            patch.setattr(runner.fwd_graph, "replay", replay)
    return engine


def _assert_same_mtp(reference, actual, depth):
    expected = {key: logits for key, _, _, logits in reference["mtp"]}
    compared = {d: 0 for d in range(depth)}
    shapes = {}
    for key, requests, shape, logits in actual["mtp"]:
        assert key in expected, ("MTP target history changed", key)
        assert torch.equal(expected[key], logits), ("MTP target logits changed", key, shape)
        compared[key[0]] += 1
        shapes.setdefault(requests, set()).add(shape[0])
    assert all(compared.values()), ("an MTP depth had no same-history comparison", compared)
    return compared, {requests: sorted(rows) for requests, rows in shapes.items()}


MAMBA_CASES = [
    bi.Case("mamba-partial", prompt_length=129),
    bi.Case(
        "mamba-chunked",
        prompt_length=401,
        context={"enable_chunked_prefill": True, "max_tokens": 256},
    ),
    bi.Case("mamba-graph", prompt_length=129, context={"num_cuda_graphs": 4}),
    bi.Case("mamba-tp2", prompt_length=129, tp=2),
]


@pytest.mark.parametrize("case", MAMBA_CASES, ids=lambda case: case.name)
def test_mamba_dynamic_batch_invariance(case):
    engines = []
    with bi.invariant_runtime(case) as (backend, version), pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            bi,
            "build_engine",
            partial(_build_model_engine, mamba=True, engines=engines, patch=patch),
        )
        reference = bi.run_order(case, backend, version, "solo")
        actual = bi.run_order(case, backend, version, "back")
        bi.assert_same_target(reference, actual)
        expected = {end: state for end, _, _, _, state in engines[0]._bi_model_evidence["states"]}
        compared = 0
        slots = set()
        for end, decode, slot, _, state in engines[1]._bi_model_evidence["states"]:
            assert state.dtype == torch.float32
            slots.add(slot)
            if end in expected:
                assert torch.equal(state, expected[end]), ("Mamba target state changed", end)
                compared += int(decode)
        assert compared > 0, "no target decode-state comparison"
        assert any(s["decode"] and s["physical"] == 64 for s in reference[1].steps)
        assert any(
            s["decode"] and s["requests"] == 65 and s["physical"] == 128 for s in actual[1].steps
        )
        print(
            f"BI_MAMBA_WITNESS {case.name} {backend} {version} "
            f"decode_state_comparisons {compared} slots {sorted(slots)} physical {[64, 128]}"
        )


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("graph_mode", ["eager", "decode", "mixed"])
def test_mtp_dynamic_batch_invariance(depth, graph_mode):
    context = dict(num_speculative_tokens=depth)
    if graph_mode != "eager":
        context.update(num_cuda_graphs=-1, max_tokens=256)
    if graph_mode == "mixed":
        context.update(
            enable_chunked_prefill=True,
            use_cuda_graphs_for_non_decode_steps=True,
            cuda_graph_mixed_prefill_count=2,
            cuda_graph_all_prefills=True,
        )
    case = bi.Case(
        "mtp",
        context=context,
        sampling={"num_tokens_to_generate": 10},
        prompt_length=401 if graph_mode == "mixed" else 17,
    )
    engines = []
    with bi.invariant_runtime(case) as (backend, version), pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            bi,
            "build_engine",
            partial(_build_model_engine, mamba=False, engines=engines, patch=patch),
        )
        reference = bi.run_order(case, backend, version, "solo")
        actual = bi.run_order(case, backend, version, "staggered")
        bi.assert_same_target(reference, actual)
        evidence = engines[1]._bi_model_evidence
        compared, mtp_shapes = _assert_same_mtp(engines[0]._bi_model_evidence, evidence, depth)
        print("BI_MTP_LOGIT_WITNESS", depth, graph_mode, compared, mtp_shapes)
        if graph_mode == "eager":
            assert evidence["inner_calls"] == set(range(depth)), "missing actual target MTP depth"
        else:
            assert evidence["replayed_depths"] == set(range(depth)), "missing captured MTP replay"
        expected_widths = {
            n: (
                (n + 3) // 4 * 4 * (depth + 1)
                if graph_mode == "eager"
                else ((n * (depth + 1) + 63) // 64) * 64
            )
            for n in (1, 65, 64)
        }
        assert any(s["decode"] and s["physical"] == expected_widths[1] for s in reference[1].steps)
        transitions = [
            (i, s["requests"])
            for i, s in enumerate(actual[1].steps)
            if s["decode"] and s["physical"] == expected_widths.get(s["requests"])
        ]
        wide = next((i for i, requests in transitions if requests == 65), None)
        narrow = next(
            (i for i, requests in transitions if requests == 64 and wide is not None and i > wide),
            None,
        )
        assert wide is not None and narrow is not None, ("missing 65-to-64 transition", transitions)
        assert all(requests in mtp_shapes for requests in (65, 64))
        print(
            f"BI_MTP_WITNESS {(depth, graph_mode, backend, version)} comparisons {compared} "
            f"verifier_physical {expected_widths} mtp_rows {mtp_shapes} "
            f"replayed_depths {evidence['replayed_depths']}"
        )
