# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare teacher-forced training logprobs with paged SDSA prefill and decode."""

import json

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant import dsa_gqa_decode_kernel
from megatron.core.transformer.experimental_attention_variant.dsa_layer_specs import dsa_stack_spec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not dsa_gqa_decode_kernel.HAVE_CUTEDSL, reason="CuTeDSL is required"
)


@pytest.fixture(autouse=True)
def initialize_logprob_test_distributed():
    Utils.initialize_distributed()
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("backend", ["reference", "min-memory-triton"])
@pytest.mark.parametrize("topk", [8, 512])
@pytest.mark.parametrize(
    "head_dim,index_head_dim,rotary_base,rotary_percent",
    [(128, 128, 10000, 1.0), (192, 128, 500000, 0.5), (256, 64, 1000000, 0.5)],
)
def test_decode_logprobs_match_training(
    monkeypatch, backend, topk, head_dim, index_head_dim, rotary_base, rotary_percent
):
    """The same weights and token histories give close full-vocabulary logprobs."""
    torch.manual_seed(53)
    vocab_size, decode_steps = 1024, 8
    prompt_lengths = (31, 63, 513)
    config = TransformerConfig(
        num_layers=4,
        hidden_size=512,
        ffn_hidden_size=1024,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=head_dim,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        is_hybrid_model=True,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_kernel_backend=backend,
        dsa_indexer_n_heads=1,
        dsa_indexer_head_dim=index_head_dim,
        dsa_indexer_topk=topk,
        dsa_indexer_loss_coeff=0.1,
    )
    # Match the model-wide settings passed to the indexer by argument_utils.
    config.rope_type = "rope"
    config.rotary_base = rotary_base
    config.rotary_percent = rotary_percent
    model = HybridModel(
        config=config,
        hybrid_stack_spec=dsa_stack_spec,
        vocab_size=vocab_size,
        max_sequence_length=640,
        hybrid_layer_pattern="*-*-",
        position_embedding_type="rope",
        rotary_base=rotary_base,
        rotary_percent=rotary_percent,
        parallel_output=False,
    ).cuda()
    tokens = [
        torch.randint(vocab_size, (length + decode_steps + 1,), device="cuda")
        for length in prompt_lengths
    ]

    # Exercise the actual training branch: gradients enabled, dropout disabled,
    # indexer loss enabled, and no inference context. Causal attention lets one
    # teacher-forced forward supply references for every later decode step.
    model.train()
    training_logprobs = []
    with torch.enable_grad():
        for sequence in tokens:
            inputs = sequence[:-1].unsqueeze(0)
            logits = model(
                input_ids=inputs,
                position_ids=torch.arange(inputs.size(1), device="cuda").unsqueeze(0),
                attention_mask=None,
                runtime_gather_output=True,
            )
            assert logits.requires_grad
            training_logprobs.append(logits.detach().float().log_softmax(-1)[0])
            del logits

    context = DynamicInferenceContext(
        model_config=config,
        inference_config=InferenceConfig(
            max_sequence_length=640,
            max_requests=3,
            max_tokens=1024,
            block_size_tokens=64,
            buffer_size_gb=0.05,
            unified_memory_level=0,
            num_cuda_graphs=None,
            materialize_only_last_token_logits=True,
            use_flashinfer_fused_rope=False,
        ),
    )
    for request_id, length in enumerate(prompt_lengths):
        context.add_request(
            DynamicInferenceRequest(
                request_id=request_id,
                prompt_tokens=tokens[request_id][:length].cpu(),
                sampling_params=SamplingParams(num_tokens_to_generate=decode_steps + 1),
            )
        )

    decode_calls = []
    original_decode = dsa_gqa_decode_kernel.dsa_gqa_decode_attention

    def capture_decode(*args, **kwargs):
        decode_calls.append(1)
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(dsa_gqa_decode_kernel, "dsa_gqa_decode_attention", capture_decode)
    model.eval()
    errors, target_errors = [], []
    with torch.inference_mode(), InferenceMode.active():
        for step in range(decode_steps + 1):
            context.initialize_attention_state()
            input_ids, position_ids = context.current_input_and_position_ids()
            # Allocating a new KV page can reorder active requests. Align both
            # references and teacher-forced tokens with the current request IDs.
            assert context.paused_request_count == 0
            request_ids = context.request_ids[: context.total_request_count].tolist()
            if step:
                expected_positions = torch.tensor(
                    [prompt_lengths[request_id] - 1 + step for request_id in request_ids],
                    device="cuda",
                )
                expected_inputs = torch.stack(
                    [
                        tokens[request_id][prompt_lengths[request_id] - 1 + step]
                        for request_id in request_ids
                    ]
                )
                torch.testing.assert_close(position_ids[0], expected_positions)
                torch.testing.assert_close(input_ids[0], expected_inputs)
            logits = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                inference_context=context,
                runtime_gather_output=True,
            )
            assert logits.shape == (1, len(prompt_lengths), vocab_size)
            actual = logits[0].float().log_softmax(-1)
            expected = torch.stack(
                [
                    training_logprobs[request_id][prompt_lengths[request_id] - 1 + step]
                    for request_id in request_ids
                ]
            )
            delta = (actual - expected).abs()
            errors.append(delta)
            targets = torch.stack(
                [
                    tokens[request_id][prompt_lengths[request_id] + step]
                    for request_id in request_ids
                ]
            )
            target_errors.append(delta.gather(1, targets[:, None]).squeeze(1))
            if step < decode_steps:
                # Feed the training tokens, so sampling cannot change the histories.
                context.update_requests(
                    active_requests_mask=torch.ones(len(tokens), dtype=torch.int32),
                    new_tokens=targets.cpu(),
                )

    assert len(decode_calls) == 2 * decode_steps
    errors = torch.stack(errors)
    target_errors = torch.stack(target_errors)
    metrics = {
        "backend": backend,
        "topk": topk,
        "head_dim": head_dim,
        "index_head_dim": index_head_dim,
        "rotary_base": rotary_base,
        "rotary_percent": rotary_percent,
        "prefill_max_abs": errors[0].max().item(),
        "decode_max_abs": errors[1:].max().item(),
        "decode_mean_abs": errors[1:].mean().item(),
        "target_max_abs": target_errors.max().item(),
    }
    print("SDSA_LOGPROB " + json.dumps(metrics, sort_keys=True))
    # Absolute errors are in nats: 0.02 corresponds to about 2% probability error.
    assert errors.max().item() <= 2e-2, metrics
    assert errors.mean().item() <= 2e-3, metrics
