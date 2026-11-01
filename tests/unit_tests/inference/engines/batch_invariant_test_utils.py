# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real-engine batch-invariance fixtures; backend groups run in fresh interpreters."""

import gc
import inspect
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.config import AsyncScheduleMode, InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_inference_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import attention
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from megatron.core.transformer.enums import AttnBackend, InferenceCudaGraphScope
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from tests.unit_tests.inference.test_data_parallel_inference_coordinator import (
    DummyTokenizer as _DummyTokenizer,
)
from tests.unit_tests.test_utilities import Utils

TARGET = 101
VOCAB = 128


class DummyTokenizer(_DummyTokenizer):
    """Reuse the coordinator's integer tokenizer with score-offset support."""

    @staticmethod
    def offsets(tokens, text):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        result = []
        cursor = 0
        for token in tokens:
            result.append(cursor)
            cursor += len(str(token)) + 1
        return result


@dataclass(frozen=True)
class Case:
    """One supported feature interaction, not an environment fallback recipe."""

    name: str
    model: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    sampling: dict = field(default_factory=dict)
    prompt_length: int = 17
    warm_prefix: bool = False
    tp: int = 1
    pp: int = 1
    sp: bool = False


MLA_CASE = Case(
    "mla",
    model=dict(
        multi_latent_attention=True,
        cache_mla_latents=True,
        num_attention_heads=64,
        qk_layernorm=True,
    ),
    context={"block_size_tokens": 64},
)


@contextmanager
def invariant_runtime(case):
    """Pin process-global mode; launch each selected backend in a fresh process."""
    backend = os.environ.get("MCORE_BI_TEST_BACKEND", "triton")
    # Default to the newest installed FlashAttention; the variable pins one explicitly.
    fa_version = int(os.environ.get("MCORE_BI_TEST_FA_VERSION") or (4 if attention.HAVE_FA4 else 3))
    assert backend in ("triton", "te_native", "deepgemm")
    assert not bik.is_batch_invariant_mode_enabled(), "inherited a live backend from another test"
    if backend == "te_native":
        assert (
            os.environ.get("CUBLASLT_WORKSPACE_SIZE") == "0"
        ), "Set CUBLASLT_WORKSPACE_SIZE=0 before starting the distributed interpreter"
    assert fa_version in (3, 4)
    assert attention.HAVE_FA3 if fa_version == 3 else attention.HAVE_FA4
    assert bik.te_supports_batch_invariant_attention()
    # Another test module can change these class attributes during collection.
    rounders = (DynamicInferenceContext.TOKEN_ROUNDER, DynamicInferenceContext.REQUEST_ROUNDER)
    DynamicInferenceContext.TOKEN_ROUNDER = 64
    DynamicInferenceContext.REQUEST_ROUNDER = 4
    deterministic = torch.are_deterministic_algorithms_enabled()
    bik.enable_batch_invariant_mode(backend=backend, collective="ordered")
    try:
        torch.use_deterministic_algorithms(deterministic or case.name.startswith("mamba"))
        Utils.initialize_model_parallel(
            case.tp,
            case.pp,
            expert_model_parallel_size=case.model.get("expert_model_parallel_size", 1),
        )
        with pytest.MonkeyPatch.context() as patch:
            for name, value in (("FLASH_ATTN", "1"), ("FUSED_ATTN", "0"), ("UNFUSED_ATTN", "0")):
                patch.setenv("NVTE_" + name, value)
            yield backend, fa_version
    finally:
        torch.use_deterministic_algorithms(deterministic)
        if case.model.get("expert_model_parallel_size", 1) > 1:
            NVLSAllGatherVDispatcher._delete_buffers()
            SymmetricMemoryManager.destroy()
        InferenceMode.unset_active()
        bik.disable_batch_invariant_mode()
        DynamicInferenceContext.TOKEN_ROUNDER, DynamicInferenceContext.REQUEST_ROUNDER = rounders
        gc.collect()
        torch.cuda.empty_cache()
        Utils.destroy_model_parallel()


def model_defaults(backend, fa_version):
    return dict(
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        batch_invariant_backend=backend,
        flash_attention_version=fa_version,
        normalization="RMSNorm",
        params_dtype=torch.bfloat16,
        bf16=True,
        attention_backend=AttnBackend.flash,
        transformer_impl="transformer_engine",
        nccl_all_reduce_for_prefill=False,
        inference_rng_tracker=True,
        inference_sampling_seed=333,
    )


def build_engine(case, backend, fa_version):
    """Build the same tiny, real model for every ordering of a case."""
    torch.manual_seed(321)
    model_parallel_cuda_manual_seed(
        321, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
    )
    graphed = case.context.get("num_cuda_graphs") is not None
    model_options = dict(
        **model_defaults(backend, fa_version),
        num_layers=2 if case.pp == 1 else 4,
        hidden_size=128,
        num_attention_heads=4,
        tensor_model_parallel_size=case.tp,
        pipeline_model_parallel_size=case.pp,
        sequence_parallel=case.sp,
        pipeline_dtype=torch.bfloat16,
        cuda_graph_impl="local" if graphed else "none",
        inference_cuda_graph_scope=(
            InferenceCudaGraphScope.block if graphed else InferenceCudaGraphScope.none
        ),
    )
    model_options.update(case.model)
    cfg = (
        MLATransformerConfig if model_options.get("multi_latent_attention") else TransformerConfig
    )(**model_options)
    factories = {
        "transformer_engine": get_gpt_layer_with_transformer_engine_spec,
        "local": get_gpt_layer_local_spec,
        "inference_optimized": get_gpt_layer_with_inference_spec,
    }
    spec_options = {"normalization": cfg.normalization} if cfg.transformer_impl == "local" else {}
    spec_options["num_experts"] = cfg.num_moe_experts
    if cfg.multi_latent_attention:
        spec_options.update(multi_latent_attention=True, qk_layernorm=cfg.qk_layernorm)
    model = (
        GPTModel(
            config=cfg,
            transformer_layer_spec=factories[cfg.transformer_impl](**spec_options),
            vocab_size=VOCAB,
            max_sequence_length=1024,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            position_embedding_type="rope",
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        .cuda()
        .eval()
    )
    model = Float16Module(cfg, model).eval()
    options = dict(
        max_sequence_length=1024,
        buffer_size_gb=0.125,
        block_size_tokens=256,
        max_requests=128,
        max_tokens=512,
        num_cuda_graphs=None,
        materialize_only_last_token_logits=False,
        use_cuda_graphs_for_non_decode_steps=False,
        unified_memory_level=0,
        async_sched_mode=AsyncScheduleMode.LEGACY,
    )
    options.update(case.context)
    ctx = DynamicInferenceContext(cfg, InferenceConfig(**options))
    tokenizer = DummyTokenizer(VOCAB, bos=1, eod=VOCAB - 1)
    wrapper = GPTInferenceWrapper(model, ctx)
    return DynamicInferenceEngine(TextGenerationController(wrapper, tokenizer), ctx)


class ForwardWitness:
    """Capture target-owned tensors before the next step can compact or reuse them."""

    def __init__(self, engine, patch):
        self.engine = engine
        self.model_config = engine.controller.inference_wrapped_model.model.config
        self.steps, self.dummy_steps = [], []
        self.sample_steps, self.async_overlaps, self.retirement_events = [], [], []
        self.current = self._active_async_overlap = self.capturing = None
        self.captures = engine.__dict__.setdefault("_bi_graph_captures", {})
        controller, ctx = engine.controller, engine.context
        original = controller._dynamic_step_forward_logits

        def target_is_active():
            active = ctx.request_ids[ctx.paused_request_count : ctx.total_request_count]
            return TARGET in engine.requests and bool((active == TARGET).any())

        def forward(input_ids, position_ids):
            n = ctx.active_token_count
            req_idxs = ctx.token_to_request_idx[:n].long()
            ids = ctx.request_ids[req_idxs]
            rows = (ids == TARGET).nonzero().flatten()
            self.current = None
            dummy = ctx._bookkeeping_no_real_work and not ctx.is_creating_cuda_graphs
            if (
                rows.numel()
                and TARGET in engine.requests
                and not (dummy or ctx.is_creating_cuda_graphs)
            ):
                request_idx = int(req_idxs[rows[0]])
                self.current = dict(
                    positions=position_ids[0, rows].clone(),
                    tokens=input_ids[0, rows].clone(),
                    logical=n,
                    physical=int(input_ids.shape[1]),
                    decode=ctx.is_decode_only(),
                    target_query=int(ctx.request_query_lengths[request_idx]),
                    target_offset=int(ctx.request_kv_length_offsets[request_idx]),
                    requests=ctx.total_request_count - ctx.paused_request_count,
                    graph=ctx.using_cuda_graph_this_step(),
                    replay=0,
                    captured_graphs=[],
                    attention=[],
                    mla=[],
                    sinks=[],
                    gemms=[],
                    neighbor_queries={
                        rid: int(ctx.request_query_lengths[idxs[0]])
                        for rid in (201, 202)
                        if (idxs := (ctx.request_ids[: ctx.total_request_count] == rid).nonzero())
                        .flatten()
                        .numel()
                    },
                    target_row=int(rows[0]),
                )
            elif dummy:
                self.current = dict(
                    physical=input_ids.shape[1], attention=[], gemms=[], sinks=[], mla=[]
                )
            try:
                result = original(input_ids, position_ids)
                if dummy:
                    self.dummy_steps.append(self.current)
                elif self.current is not None:
                    getattr(engine, "_bi_after_forward", lambda: None)()
                    if ctx.config.materialize_only_last_token_logits:
                        mapped = ctx.active_logit_idxs[: ctx.num_last_token_logits].long()
                        selected = torch.isin(mapped, rows.to(mapped.device))
                        rows = mapped[selected]
                        logits = controller._all_logits_cuda[0, : ctx.num_last_token_logits][
                            selected.to("cuda")
                        ]
                        self.current["positions"] = position_ids[0, rows].clone()
                        self.current["tokens"] = input_ids[0, rows].clone()
                    else:
                        logits = controller._all_logits_cuda[0, rows.to("cuda")]
                    self.current["logits"] = logits.detach().clone()
                    self.steps.append(self.current)
                return result
            finally:
                self.current = None

        patch.setattr(controller, "_dynamic_step_forward_logits", forward)

        def observe(owner, name, kind):
            fn = getattr(owner, name, None)
            if fn is None:
                return

            @wraps(fn)
            def call(*args, **kwargs):
                result = fn(*args, **kwargs)
                record = (
                    self.captures[self.capturing]
                    if torch.cuda.is_current_stream_capturing()
                    else self.current
                )
                if record is not None:
                    if kind == "attention":
                        record[kind].append((name, kwargs.get("num_splits")))
                        if self.model_config.multi_latent_attention:
                            assert (kwargs["q"] if "q" in kwargs else args[0]).shape[-1] == 192
                    elif kind == "mla":
                        assert args[1].shape[1:] == (64, 576)
                        assert result[0].shape[1:] == (64, 64, 192)
                        assert result[1].shape[1:] == (64, 64, 128)
                        record[kind].append((tuple(args[1].shape), tuple(result[0].shape)))
                    else:
                        tensors = [x for x in args if isinstance(x, torch.Tensor)]
                        record[kind].append((name, tuple(tensors[0].shape) if tensors else ()))
                return result

            # Preserve CustomOpDef's _init_fn signature so production keeps FA3 kwargs such as q.
            if hasattr(fn, "_init_fn"):
                call.__signature__ = inspect.signature(fn._init_fn)
            patch.setattr(owner, name, staticmethod(call) if kind == "sinks" else call)

        for name in ("_flash_attn_forward", "flash_attn3_with_kvcache", "flash_attn4_varlen_func"):
            observe(attention, name, "attention")
        observe(MLASelfAttention, "uncompress_kv_from_cache", "mla")
        for layout in ("varlen", "bshd"):
            observe(attention.Attention, f"_apply_sink_softmax_correction_{layout}", "sinks")
        for name in ("matmul_persistent", "_mm_deepgemm"):
            observe(bik, name, "gemms")
        # Native TE binds GEMM locally; the other dense providers use torch.matmul.
        if bik.get_batch_invariant_backend() == "te_native":
            from transformer_engine.pytorch.module import layernorm_linear, linear

            observe(torch, "matmul", "gemms")
            observe(linear, "general_gemm", "gemms")
            observe(layernorm_linear, "general_gemm", "gemms")
        async_overlap = controller._run_async_sched_step_overlap
        async_forward = controller._run_async_sched_forward

        @wraps(async_overlap)
        async def observe_async_overlap(**kwargs):
            record = dict(
                target_active=target_is_active(),
                pending=controller._async_sched_forward.is_valid,
                forwards=0,
                returned_previous=False,
            )
            self._active_async_overlap = record
            try:
                result = await async_overlap(**kwargs)
                record["returned_previous"] = result.output is not None
                return result
            finally:
                self._active_async_overlap = None
                if record["target_active"]:
                    self.async_overlaps.append(record)

        @wraps(async_forward)
        def observe_async_forward(*args, **kwargs):
            if self._active_async_overlap is not None:
                self._active_async_overlap["forwards"] += 1
            return async_forward(*args, **kwargs)

        patch.setattr(controller, "_run_async_sched_step_overlap", observe_async_overlap)
        patch.setattr(controller, "_run_async_sched_forward", observe_async_forward)
        capture_begin = torch.cuda.CUDAGraph.capture_begin

        def graph_begin(graph, *args, **kwargs):
            result = capture_begin(graph, *args, **kwargs)
            self.capturing = graph
            ctx._bi_capturing_graph = graph
            self.captures[graph] = dict(attention=[], gemms=[], sinks=[], mla=[])
            return result

        patch.setattr(torch.cuda.CUDAGraph, "capture_begin", graph_begin)
        replay = torch.cuda.CUDAGraph.replay

        def graph_replay(graph):
            result = replay(graph)
            if self.current is not None and "replay" in self.current:
                self.current["replay"] += 1
                if graph in self.captures:
                    if self.captures[graph]["attention"]:
                        self.current["captured_graphs"].append(id(graph))
                    for kind, calls in self.captures[graph].items():
                        self.current[kind].extend(calls)
            return result

        patch.setattr(torch.cuda.CUDAGraph, "replay", graph_replay)
        sampler = controller._sampling
        sample_kernel = sampler.sample_kernel

        @wraps(sample_kernel)
        def sample(logits, n, context, **kwargs):
            result = sample_kernel(logits, n, context, **kwargs)
            ids = context.request_ids[context.paused_request_count : context.total_request_count]
            target_rows = (ids == TARGET).nonzero().flatten()
            if target_rows.numel() and kwargs.get("token_to_request_index") is None:
                target_row = int(target_rows[0])
                indices = kwargs.get("gather_indices")
                selected = logits[:n] if indices is None else logits[indices[:n].long()]
                processed = sampler.log_probs_kernel(selected, context)
                token = result[target_row].clone()
                self.sample_steps.append(
                    dict(
                        logits=selected[target_row].detach().clone(),
                        log_probs=processed[target_row].detach().clone(),
                        token=token,
                        sampler=type(sampler).__name__,
                        requests=n,
                        filters=context.active_sampling_filter_flags(n),
                    )
                )
            return result

        patch.setattr(sampler, "sample_kernel", sample)

    def assert_active(self, fa_version):
        """Reject configured-only backends and observations from other requests."""
        assert self.steps, "target never executed a forward"
        calls = [call for step in self.steps for call in step["attention"]]
        assert calls, "no real target-containing attention call"
        expected = (
            {"_flash_attn_forward", "flash_attn3_with_kvcache"}
            if fa_version == 3
            else {"flash_attn4_varlen_func"}
        )
        assert all(name in expected and splits == 1 for name, splits in calls), calls
        assert any(step["gemms"] for step in self.steps), "no real target-containing GEMM call"
        if self.model_config.softmax_type != "vanilla":
            assert any(s["sinks"] for s in self.steps), "target never executed sink correction"
        if self.model_config.multi_latent_attention:
            assert any(
                s["decode"] and s["mla"] for s in self.steps
            ), "no target latent-cache expansion"
        assert all(
            step["physical"] % 64 == 0
            for step in self.steps
            if not self.engine.num_speculative_tokens or step["graph"] or not step["decode"]
        )
        if self.engine.context.config.async_sched_mode == AsyncScheduleMode.ASYNC:
            assert any(
                item["target_active"]
                and item["pending"]
                and item["forwards"] == 1
                and item["returned_previous"]
                for item in self.async_overlaps
            ), "target never consumed pending logits while launching an overlapping forward"


def target_prompt(length):
    return [(7 + i * 13) % (VOCAB - 2) + 1 for i in range(length)]


def run_order(case, backend, fa_version, order, *, sampling=None):
    """Run identical target histories alone, moved in a batch, or amid arrivals."""
    with pytest.MonkeyPatch.context() as construction_patch:
        create_graphs = DynamicInferenceEngine.create_cuda_graphs

        def capture(engine, *args, **kwargs):
            ForwardWitness(engine, construction_patch)
            return create_graphs(engine, *args, **kwargs)

        construction_patch.setattr(DynamicInferenceEngine, "create_cuda_graphs", capture)
        engine = build_engine(case, backend, fa_version)
    params = dict(
        num_tokens_to_generate=6,
        top_k=1,
        termination_id=-1,
        return_log_probs=True,
        return_prompt_tokens=True,
    )
    params.update({**case.sampling, **(sampling or {})})
    target_params = SamplingParams(**params)
    depth = case.context.get("num_speculative_tokens", 0)
    short_neighbor_length = depth + 2
    target = target_prompt(case.prompt_length)
    finished = {}
    witness = None

    def step():
        result = engine.step_modern()
        for req in result["finished_requests"]:
            req.finalize_text(engine.controller.tokenizer)
            finished[req.request_id] = req
            if witness is not None and req.request_id == 201:
                witness.retirement_events.append(
                    dict(
                        target_active=TARGET in engine.requests, generated=len(req.generated_tokens)
                    )
                )

    def drain():
        for _ in range(256):
            if not engine.has_unfinished_requests():
                break
            step()
        assert not engine.has_unfinished_requests(), "bounded engine drain exhausted"

    if case.warm_prefix:
        engine.add_request(
            99, target, SamplingParams(num_tokens_to_generate=1, top_k=1, termination_id=-1)
        )
        drain()
        assert engine.context.kv_block_allocator.enable_prefix_caching
    with pytest.MonkeyPatch.context() as patch:
        witness = ForwardWitness(engine, patch)
        neighbors = list(range(201, 265))

        def add_neighbors(*, retire_early=False):
            for rid in neighbors:
                # Stochastic targets share a real heterogeneous launch with top-k and top-p peers.
                filters = dict(top_k=1)
                if target_params.top_k != 1:
                    filters = dict(top_k=7) if rid % 2 else dict(top_p=0.8)
                output_length = (
                    short_neighbor_length
                    if retire_early and rid == 201
                    else 9 if rid == 202 or depth else 6
                )
                neighbor = SamplingParams(
                    num_tokens_to_generate=output_length, termination_id=-1, **filters
                )
                prompt = target_prompt(33) if rid == 202 else [rid % (VOCAB - 1)]
                engine.add_request(rid, prompt, neighbor)

        if order == "back":
            add_neighbors()
        engine.add_request(TARGET, target, target_params)
        if order == "staggered":
            for _ in range(256):
                if engine.get_request(TARGET).generated_tokens:
                    break
                step()
            assert (
                1 <= len(engine.get_request(TARGET).generated_tokens) <= depth + 1
            ), "target did not finish its first sampling step before staggered arrivals"
            assert TARGET not in finished, "target retired before staggered arrivals"
            add_neighbors(retire_early=True)
        elif order == "front":
            add_neighbors()
        else:
            assert order in ("solo", "back")
        drain()
        witness.assert_active(fa_version)
        assert TARGET in finished
        # Observe retirement before cleanup can make a leaking run appear green.
        assert engine.context.total_request_count == 0
        assert engine.context.kv_block_allocator.get_active_used() == 0
        if not case.warm_prefix:
            assert engine.context.kv_block_allocator.get_total_used() == 0
        if case.context.get("num_cuda_graphs") is not None:
            assert any(
                s["graph"] and s["captured_graphs"] for s in witness.steps
            ), "target never replayed a graph with observed captured attention"
        if case.context.get("use_cuda_graphs_for_non_decode_steps"):
            assert any(
                not s["decode"] and s["graph"] and s["captured_graphs"] for s in witness.steps
            ), "no target nondecode replay with captured attention provenance"
        if case.context.get("enable_chunked_prefill"):
            chunks = [s for s in witness.steps if int(s["positions"][0]) < case.prompt_length - 1]
            assert len(chunks) >= 2, "target did not execute multiple prefill chunks"
        if case.warm_prefix:
            assert (
                min(int(s["positions"].min()) for s in witness.steps) >= 256
            ), "target did not reuse its prefix"
        if order == "staggered":
            assert any(
                event["target_active"] and event["generated"] == short_neighbor_length
                for event in witness.retirement_events
            ), "short neighbor did not retire while the target remained active"
    return finished[TARGET], witness


def assert_same_target(reference, actual, *, require_trajectory=True):
    """Check every shared target position, never unrelated padded rows."""
    ref_req, ref_witness = reference
    req, witness = actual
    if require_trajectory:
        for field_name in (
            "generated_tokens",
            "prompt_log_probs",
            "generated_log_probs",
            "prompt_top_n_logprobs",
            "generated_top_n_logprobs",
        ):
            assert getattr(req, field_name, None) == getattr(ref_req, field_name, None), field_name
    expected = {}
    for step in ref_witness.steps:
        for pos, token, logits in zip(step["positions"], step["tokens"], step["logits"]):
            key = int(pos), int(token)
            if key in expected:
                assert torch.equal(expected[key], logits), ("reference repeated position", key)
            expected[key] = logits
    comparisons = 0
    for step in witness.steps:
        for pos, token, logits in zip(step["positions"], step["tokens"], step["logits"]):
            key = int(pos), int(token)
            assert key in expected, ("target history changed", key)
            assert torch.equal(expected[key], logits), (
                "target logits changed",
                key,
                step["physical"],
            )
            comparisons += 1
    assert comparisons > 0
