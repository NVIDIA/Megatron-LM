# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    set_batch_invariant_mode,
    te_supports_batch_invariant_attention,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.rl.rl_utils import selective_log_softmax
from tests.unit_tests.test_utilities import Utils

try:
    from flash_attn_3.flash_attn_interface import _flash_attn_forward
    from flash_attn_3.flash_attn_interface import (
        flash_attn_with_kvcache as flash_attn3_with_kvcache,
    )

    HAVE_FA3 = True
except ImportError:
    HAVE_FA3 = False

try:
    # Blackwell (e.g. GB200) ships FlashAttention-4 instead of FA3; the batch-invariant
    # attention paths honor config.flash_attention_version, so these tests run there too.
    from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen_func  # noqa: F401

    HAVE_FA4 = True
except ImportError:
    HAVE_FA4 = False

# Batch-invariant mode requires an explicit FlashAttention version; pick the newest
# one available so training and inference run the same kernel.
_BIK_FA_VERSION = 4 if HAVE_FA4 else 3
pytestmark = pytest.mark.skipif(
    not te_supports_batch_invariant_attention(),
    reason="Batch-invariant attention requires TransformerEngine PR #3204 or >= 2.18.",
)


class DummyTokenizer:
    def __init__(self, vocab_size: int, bos: int | None = None, eod: int = 0, pad: int = 0):
        self.vocab_size = vocab_size
        self.bos = bos
        self.eod = eod
        self.pad = pad

    def tokenize(self, prompt):
        if isinstance(prompt, str):
            tokens = [int(tok) % self.vocab_size for tok in prompt.strip().split()]
        else:
            tokens = list(prompt)
        return tokens

    def detokenize(self, tokens, skip_special_tokens: bool = False):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if skip_special_tokens and self.eod in tokens:
            tokens = [tok for tok in tokens if tok != self.eod]
        return " ".join(str(tok) for tok in tokens)

    def offsets(self, tokens, text):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        offsets = []
        cursor = 0
        for tok in tokens:
            offsets.append(cursor)
            cursor += len(str(tok)) + 1
        return offsets


def _configure_flash_attention_env():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ['NVTE_FUSED_ATTN'] = '0'
    os.environ['NVTE_FLASH_ATTN'] = '1'
    os.environ['NVTE_UNFUSED_ATTN'] = '0'


def _build_flash_attn_bik_model(seq_len: int, vocab_size: int, hidden_size: int = 128) -> GPTModel:
    cfg = TransformerConfig(
        num_layers=2,
        hidden_size=hidden_size,
        num_attention_heads=4,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        flash_attention_version=_BIK_FA_VERSION,
        normalization="RMSNorm",
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.flash,
        fp32_residual_connection=False,
        nccl_all_reduce_for_prefill=False,
    )
    cfg.fp16 = False
    cfg.bf16 = True
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
    )
    return model.cuda().eval()


def _train_forward_logprobs(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = tokens.shape
    position_ids = (
        torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, seq_len)
    )
    attention_mask = torch.ones(
        batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=tokens.device
    )
    with torch.no_grad():
        # runtime_gather_output matches rl_utils.get_logprobs; without it the model
        # asserts once it has served inference requests (in-inference-mode postprocess).
        logits = model(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            runtime_gather_output=True,
        )
    logprobs = selective_log_softmax(logits[:, :-1, :], tokens[:, 1:])
    return logprobs


@pytest.mark.skipif(
    not (is_te_min_version("2.10.0") and (HAVE_FA3 or HAVE_FA4)),
    reason="TestGPTModelBatchInvariant requires TE >= 2.10.0 and FlashAttention-3 or -4",
)
class TestGPTModelBatchInvariant:
    """End-to-end batch-invariance tests for GPT."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _configure_flash_attention_env()
        model_parallel_cuda_manual_seed(321)
        self.sequence_length = 32
        self.vocab_size = 96

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_forward_batch_invariant(self):
        _configure_flash_attention_env()
        model = _build_flash_attn_bik_model(self.sequence_length, self.vocab_size)
        model = Float16Module(model.config, model).eval()
        batch_size = 6
        splits = [2, 1, 3]
        input_ids = torch.randint(
            low=0, high=self.vocab_size, size=(batch_size, self.sequence_length), device="cuda"
        )
        position_ids = (
            torch.arange(self.sequence_length, device="cuda").unsqueeze(0).repeat(batch_size, 1)
        )
        attention_mask = torch.ones(
            batch_size,
            1,
            self.sequence_length,
            self.sequence_length,
            dtype=torch.bool,
            device="cuda",
        )

        with set_batch_invariant_mode(True):
            with torch.no_grad():
                logits_full = model(
                    input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
                ).to(torch.float32)
                chunked_logits = []
                start = 0
                for split in splits:
                    end = start + split
                    chunked_logits.append(
                        model(
                            input_ids=input_ids[start:end],
                            position_ids=position_ids[start:end],
                            attention_mask=attention_mask[start:end],
                        ).to(torch.float32)
                    )
                    start = end
                logits_chunked = torch.cat(chunked_logits, dim=0)

        assert torch.equal(logits_full, logits_chunked)

    def test_dynamic_engine_matches_batched_forward_rl(self):
        _configure_flash_attention_env()
        seq_len = 48
        vocab_size = 96
        base_model = _build_flash_attn_bik_model(seq_len, vocab_size)
        inference_model = Float16Module(base_model.config, base_model).cuda().eval()

        ctx = DynamicInferenceContext(
            model_config=base_model.config,
            inference_config=InferenceConfig(
                max_sequence_length=seq_len,
                buffer_size_gb=0.125,
                block_size_tokens=16,
                num_cuda_graphs=None,
                materialize_only_last_token_logits=False,
                use_cuda_graphs_for_non_decode_steps=False,
                unified_memory_level=0,
            ),
        )

        wrapper = GPTInferenceWrapper(inference_model, ctx)
        tokenizer = DummyTokenizer(vocab_size=vocab_size, bos=None, eod=vocab_size - 1, pad=0)
        controller = TextGenerationController(wrapper, tokenizer)
        engine = DynamicInferenceEngine(controller=controller, context=ctx)

        base_vals = [3, 15, 27, 39]
        lengths = [18, 11, 23, 13]
        prompts = []
        for base, length in zip(base_vals, lengths):
            seq = [(base + i) % (vocab_size - 1) for i in range(length - 1)]
            seq.append(tokenizer.eod)
            prompts.append(seq)

        sampling_params = SamplingParams(
            num_tokens_to_generate=6,
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            add_BOS=False,
            return_log_probs=True,
            termination_id=tokenizer.eod,
        )

        finished_requests = []
        with set_batch_invariant_mode(True):
            for request_id, prompt in enumerate(prompts, start=1):
                engine.add_request(request_id, prompt, sampling_params)
            while engine.has_unfinished_requests():
                result = engine.step_modern()
                finished_requests.extend(
                    r.merge(engine.controller.tokenizer) for r in result["finished_request_records"]
                )

            assert finished_requests, "Dynamic engine did not produce any completed requests."

            for req in finished_requests:
                prompt_tokens = req.prompt_tokens.tolist()
                generated_tokens = req.generated_tokens
                full_sequence = torch.tensor(
                    prompt_tokens + generated_tokens, dtype=torch.long, device="cuda"
                ).unsqueeze(0)
                baseline_log_probs = _train_forward_logprobs(
                    inference_model, full_sequence
                ).squeeze(0)
                inference_log_probs = torch.tensor(
                    req.prompt_log_probs + req.generated_log_probs,
                    dtype=baseline_log_probs.dtype,
                    device="cuda",
                )
                assert torch.equal(
                    inference_log_probs, baseline_log_probs
                ), "Log probabilities from dynamic engine did not match batched forward."

    def test_dynamic_engine_is_batch_invariant(self):
        """Check that the dynamic engine itself is batch invariant: changing the
        order in which requests are added does not change per-request outputs."""
        _configure_flash_attention_env()
        seq_len = 48
        vocab_size = 96
        base_model = _build_flash_attn_bik_model(seq_len, vocab_size)
        inference_model = Float16Module(base_model.config, base_model).cuda().eval()

        def _run_engine_with_order(order):
            ctx = DynamicInferenceContext(
                model_config=base_model.config,
                inference_config=InferenceConfig(
                    max_sequence_length=seq_len,
                    buffer_size_gb=0.125,
                    block_size_tokens=16,
                    num_cuda_graphs=None,
                    materialize_only_last_token_logits=False,
                    use_cuda_graphs_for_non_decode_steps=False,
                    unified_memory_level=0,
                ),
            )

            wrapper = GPTInferenceWrapper(inference_model, ctx)
            tokenizer = DummyTokenizer(vocab_size=vocab_size, bos=None, eod=vocab_size - 1, pad=0)
            controller = TextGenerationController(wrapper, tokenizer)
            engine = DynamicInferenceEngine(controller=controller, context=ctx)

            base_vals = [3, 15, 27, 39]
            lengths = [18, 11, 23, 13]
            prompts = []
            for base, length in zip(base_vals, lengths):
                seq = [(base + i) % (vocab_size - 1) for i in range(length - 1)]
                seq.append(tokenizer.eod)
                prompts.append(seq)

            sampling_params = SamplingParams(
                num_tokens_to_generate=6,
                temperature=1.0,
                top_k=1,
                top_p=0.0,
                add_BOS=False,
                return_log_probs=True,
                termination_id=tokenizer.eod,
            )

            finished_by_id = {}
            with set_batch_invariant_mode(True):
                for request_id in order:
                    engine.add_request(request_id, prompts[request_id - 1], sampling_params)
                while engine.has_unfinished_requests():
                    result = engine.step_modern()
                    for r in result["finished_request_records"]:
                        req = r.merge(engine.controller.tokenizer)
                        finished_by_id[req.request_id] = req

            return finished_by_id

        # Run once with requests added in order 1,2,3,4...
        num_requests = 4
        order1 = list(range(1, num_requests + 1))
        results1 = _run_engine_with_order(order1)

        # Run again with the same requests but added in reverse order.
        order2 = list(reversed(order1))
        results2 = _run_engine_with_order(order2)

        assert set(results1.keys()) == set(results2.keys())
        for rid in results1.keys():
            r1 = results1[rid]
            r2 = results2[rid]
            assert r1.prompt_tokens.tolist() == r2.prompt_tokens.tolist()
            assert r1.generated_tokens == r2.generated_tokens
            assert r1.prompt_log_probs == r2.prompt_log_probs
            assert r1.generated_log_probs == r2.generated_log_probs

    def test_dynamic_engine_is_batch_invariant_under_kv_cache_saturation(self):
        """Saturating the KV cache must not change what requests generate.

        Under memory pressure the engine queues requests it cannot admit, pauses
        active ones when the block pool cannot grow them, and evicts whatever
        overflows the paused-retention budget. An evicted request is requeued and
        re-prefilled from a checkpoint of the tokens it had already produced. That
        reshuffles which requests share a step, how wide each decode batch is, and
        whether a step is pure-decode or mixed with a prefill -- so it leaves the
        output untouched only if the model is genuinely batch invariant.

        The other KV-pressure tests assert bookkeeping: budgets respected, blocks
        returned, output lengths correct. A request that resumes from a corrupted
        KV history passes every one of those and still generates the wrong text.
        This pins the property the bookkeeping is a proxy for -- the same requests,
        run once with room to spare and once against a pool far too small to hold
        them, produce identical tokens and identical logprobs.

        The comparison is not vacuous: run with this exact schedule but without
        batch-invariant kernels, one of these eight requests diverges from the
        reference partway through its output.
        """
        _configure_flash_attention_env()
        seq_len = 1024
        vocab_size = 96
        # FlashAttention's paged KV cache requires a block size divisible by 256, so
        # this cannot be shrunk the way the non-paged tests above shrink theirs.
        block_size_tokens = 256
        num_requests = 8
        num_tokens_to_generate = 64

        base_model = _build_flash_attn_bik_model(seq_len, vocab_size)
        inference_model = Float16Module(base_model.config, base_model).cuda().eval()
        tokenizer = DummyTokenizer(vocab_size=vocab_size, bos=None, eod=vocab_size - 1, pad=0)

        # A request only pauses when it needs a *fresh* block while the pool is full,
        # so every prompt is sized to land just below a block boundary: each one fits
        # its prompt exactly, then crosses into a new block a few decode steps in.
        # Lengths are uneven so they cross on different steps rather than in lockstep.
        # Prompts comfortably inside their last block never pause at all, however
        # small the pool -- the engine simply admits fewer of them at a time.
        lengths = [250, 500, 250, 754, 500, 246, 760, 252]
        prompts = [
            [(3 + 12 * request_index + i) % (vocab_size - 1) for i in range(length)]
            for request_index, length in enumerate(lengths)
        ]

        # termination_id=-1 runs every request to its full output length, so requests
        # keep contending for the pool instead of retiring early and relieving it.
        sampling_params = SamplingParams(
            num_tokens_to_generate=num_tokens_to_generate,
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            add_BOS=False,
            return_log_probs=True,
            termination_id=-1,
        )

        def _build_context(buffer_size_gb, paused_buffer_size_gb):
            return DynamicInferenceContext(
                model_config=base_model.config,
                inference_config=InferenceConfig(
                    max_sequence_length=seq_len,
                    buffer_size_gb=buffer_size_gb,
                    paused_buffer_size_gb=paused_buffer_size_gb,
                    block_size_tokens=block_size_tokens,
                    max_requests=num_requests,
                    num_cuda_graphs=None,
                    materialize_only_last_token_logits=False,
                    use_cuda_graphs_for_non_decode_steps=False,
                    unified_memory_level=0,
                ),
            )

        def _run(ctx):
            wrapper = GPTInferenceWrapper(inference_model, ctx)
            controller = TextGenerationController(wrapper, tokenizer)
            engine = DynamicInferenceEngine(controller=controller, context=ctx)
            allocator = ctx.kv_block_allocator

            finished_by_id = {}
            max_paused_blocks = 0
            with set_batch_invariant_mode(True):
                for request_id, prompt in enumerate(prompts, start=1):
                    engine.add_request(request_id, prompt, sampling_params)
                # Bounded so a scheduling regression fails the test instead of hanging.
                for _ in range(2000):
                    if not engine.has_unfinished_requests():
                        break
                    result = engine.step_modern()
                    # Sampled at a step boundary, after the pause/resume/evict
                    # lifecycle for this step has settled.
                    max_paused_blocks = max(max_paused_blocks, allocator.get_paused_used())
                    for record in result["finished_request_records"]:
                        req = record.merge(engine.controller.tokenizer)
                        finished_by_id[req.request_id] = req

            assert not engine.has_unfinished_requests(), "engine did not drain"
            assert len(finished_by_id) == num_requests
            # No block may be leaked, in either regime.
            assert allocator.get_total_used() == 0
            return finished_by_id, max_paused_blocks, engine.evicted_request_count

        # Reference: a pool with room for every request at once, so nothing ever
        # queues, pauses, or is evicted. This is the regime every other dynamic
        # inference test runs in.
        roomy_ctx = _build_context(buffer_size_gb=0.125, paused_buffer_size_gb=None)
        block_size_bytes = roomy_ctx.block_size_bytes
        reference, roomy_paused_blocks, roomy_evictions = _run(roomy_ctx)

        # Sizing in whole blocks (+1 byte to survive the float GB round trip) keeps
        # the pool arithmetic exact. Holding every request resident to completion
        # needs 22 blocks; 10 forces sustained queuing and pausing, while still being
        # wider than the 4 blocks the longest single request needs, so no request can
        # deadlock waiting for capacity that will never exist. The 3-block paused
        # budget is smaller than what the paused set wants, so some requests are
        # retained paused and the overflow is evicted and requeued.
        saturated_ctx = _build_context(
            buffer_size_gb=(10 * block_size_bytes + 1) / 1024**3,
            paused_buffer_size_gb=(3 * block_size_bytes + 1) / 1024**3,
        )
        assert saturated_ctx.kv_block_allocator.pool_size == 10
        assert saturated_ctx.kv_block_allocator.paused_limit == 3
        saturated, saturated_paused_blocks, saturated_evictions = _run(saturated_ctx)

        # Pin the contrast, so neither half can silently drift into the other's
        # regime and make the comparison below vacuous.
        assert roomy_paused_blocks == 0 and roomy_evictions == 0, (
            "the reference run was supposed to have room to spare, but it paused "
            f"({roomy_paused_blocks} blocks) or evicted ({roomy_evictions} requests)"
        )
        assert saturated_paused_blocks > 0, "no request was ever retained paused"
        assert saturated_evictions > 0, "no request overflowed the paused budget"

        # The actual claim: pause, evict and requeue are transparent to the output.
        assert set(reference) == set(saturated)
        for request_id, expected in reference.items():
            actual = saturated[request_id]
            assert actual.prompt_tokens.tolist() == expected.prompt_tokens.tolist()
            assert (
                actual.generated_tokens == expected.generated_tokens
            ), f"request {request_id} generated different tokens once the KV cache saturated"
            assert actual.prompt_log_probs == expected.prompt_log_probs
            assert (
                actual.generated_log_probs == expected.generated_log_probs
            ), f"request {request_id} generated different logprobs once the KV cache saturated"
