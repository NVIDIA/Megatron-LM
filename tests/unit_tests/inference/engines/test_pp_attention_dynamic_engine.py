# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dynamic engine with PP attention: the served log-probs equal the training forward."""

import types
from unittest import mock

import pytest
import torch

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
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.pp.api import PPAttention
from megatron.core.transformer.pp.decode_api import PPAttentionDecoder
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

VOCAB = 128
MAX_SEQ = 1024


def _gpu_ok():
    if not torch.cuda.is_available():
        return False
    try:
        from megatron.core.transformer.pp.api import HAVE_PP_ATTENTION
    except ImportError:
        return False
    return HAVE_PP_ATTENTION and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(
    not _gpu_ok(), reason="needs a compute capability 10.x GPU, cutlass-dsl and flash-attn-4"
)


def _model(mtp_layers=0):
    cfg = TransformerConfig(
        num_layers=2,
        hidden_size=512,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=128,
        mtp_num_layers=mtp_layers or None,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        normalization="RMSNorm",
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        attention_backend=AttnBackend.pp,
        pp_attention_super_block_tiles=1,
    )
    spec = get_gpt_layer_local_spec(normalization="RMSNorm")
    mtp = get_gpt_mtp_block_spec(cfg, spec, use_transformer_engine=False) if mtp_layers else None
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=spec,
        vocab_size=VOCAB,
        max_sequence_length=MAX_SEQ,
        mtp_block_spec=mtp,
        position_embedding_type="rope",
    )
    return Float16Module(cfg, model.cuda()).eval()


def _train_logprobs(model, tokens):
    seq = tokens.shape[1]
    pos = torch.arange(seq, device=tokens.device).unsqueeze(0)
    with torch.no_grad():
        logits = model(
            input_ids=tokens, position_ids=pos, attention_mask=None, runtime_gather_output=True
        )
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    return logp.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1).squeeze(0)


def _serve(model, prompts, chunked, spec_tokens):
    ctx = DynamicInferenceContext(
        model_config=model.config,
        inference_config=InferenceConfig(
            max_sequence_length=MAX_SEQ,
            buffer_size_gb=0.25,
            block_size_tokens=256,
            max_tokens=256 if chunked else None,
            max_requests=8,
            enable_chunked_prefill=chunked,
            num_speculative_tokens=spec_tokens,
            num_cuda_graphs=None,
            materialize_only_last_token_logits=False,
            use_cuda_graphs_for_non_decode_steps=False,
            use_flashinfer_fused_rope=None,
            unified_memory_level=0,
        ),
    )
    tokenizer = types.SimpleNamespace(
        vocab_size=VOCAB, eod=VOCAB - 1, pad=0, bos=None, detokenize=lambda *a, **k: ""
    )
    controller = TextGenerationController(GPTInferenceWrapper(model, ctx), tokenizer)
    engine = DynamicInferenceEngine(controller=controller, context=ctx)
    params = SamplingParams(
        num_tokens_to_generate=8,
        temperature=1.0,
        add_BOS=False,
        return_log_probs=True,
        termination_id=-1,
    )
    for rid, prompt in enumerate(prompts, start=1):
        engine.add_request(rid, prompt, params)
    done = []
    while engine.has_unfinished_requests():
        done.extend(engine.step_modern()["finished_requests"])
    return done


class TestPPAttentionDynamicEngine:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "chunked,spec_tokens",
        [(False, 0), (True, 0), (False, 2)],
        ids=["prefill", "chunked", "verify"],
    )
    def test_served_logprobs_equal_training_forward(self, chunked, spec_tokens):
        model = _model(mtp_layers=spec_tokens)
        gen = torch.Generator().manual_seed(0)
        lengths = [300, 129, 517, 64]
        prompts = [torch.randint(0, VOCAB - 1, (n,), generator=gen).tolist() for n in lengths]
        spy = lambda cls, name: mock.patch.object(
            cls, name, autospec=True, side_effect=getattr(cls, name)
        )
        with (
            set_batch_invariant_mode(True, backend="triton"),
            spy(PPAttention, "prefill_paged") as prefill,
            spy(PPAttentionDecoder, "verify") as decode,
        ):
            done = _serve(model, prompts, chunked, spec_tokens)
            assert len(done) == len(prompts)
            assert prefill.called and decode.called, "PP decode and prefill must both run"
            for req in done:
                tokens = req.prompt_tokens.tolist() + list(req.generated_tokens)
                ref = _train_logprobs(model, torch.tensor([tokens], device="cuda"))
                served = torch.tensor(
                    list(req.prompt_log_probs) + list(req.generated_log_probs),
                    dtype=ref.dtype,
                    device="cuda",
                )
                assert served.shape == ref.shape
                assert torch.equal(served, ref), (served - ref).abs().max().item()
