# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dynamic DSA-over-GQA inference with tensor and sequence parallelism.

A small hybrid model is saved at TP1 and loaded at TP2 (sequence parallel, one KV group
replicated on both ranks, eight query heads a rank). Both serve the same requests through the
dynamic engine, and TP2 must reproduce TP1's greedy tokens and log-probs: teacher-forced prompt
log-probs, which cover prefill (prompts longer than the indexer top-k), and generated ones, which
cover decode and mixed decode/prefill steps, eager and in CUDA graphs.
"""

import types

import pytest
import torch

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.experimental_attention_variant.dsa_layer_specs import dsa_stack_spec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

VOCAB_SIZE = 1024
TOPK = 512
# Two prompts start together; the third arrives during their decode (a mixed step). Two of the
# three are longer than the top-k, so sparse selection matters in prefill and decode.
FIRST_PROMPTS = (600, 37)
LATE_PROMPT = 1100
GENERATE = 6

pytestmark = [
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
        reason="Dynamic DSA-over-GQA inference requires SM100/SM103",
    ),
    pytest.mark.skipif(Utils.world_size < 2, reason="Tensor parallelism needs two ranks"),
]


def _config(tp: int, cuda_graphs: bool, sequence_parallel: bool) -> TransformerConfig:
    config = TransformerConfig(
        num_layers=4,
        hidden_size=512,
        ffn_hidden_size=1024,
        num_attention_heads=16,
        num_query_groups=1,
        kv_channels=128,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        is_hybrid_model=True,
        use_cpu_initialization=True,
        tensor_model_parallel_size=tp,
        sequence_parallel=tp > 1 and sequence_parallel,
        inference_rng_tracker=True,
        cuda_graph_impl="local" if cuda_graphs else "none",
        inference_cuda_graph_scope=(
            InferenceCudaGraphScope.block if cuda_graphs else InferenceCudaGraphScope.none
        ),
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_indexer_n_heads=1,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=TOPK,
        dsa_indexer_loss_coeff=0.1,
    )
    # The model-wide RoPE settings that argument_utils passes to the indexer.
    config.rope_type = "rope"
    config.rotary_base = 10000
    config.rotary_percent = 1.0
    return config


def _model(config: TransformerConfig) -> HybridModel:
    model = HybridModel(
        config=config,
        hybrid_stack_spec=dsa_stack_spec,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=2048,
        hybrid_layer_pattern="*-*-",
        position_embedding_type="rope",
        rotary_base=10000,
        rotary_percent=1.0,
    ).cuda()
    for param in model.parameters():
        param.data = param.data.to(torch.bfloat16)
    return model.eval()


@torch.inference_mode()
def _serve(model, config, prompts, cuda_graphs):
    """Serve the prompts greedily; returns {request_id: request}."""
    context = DynamicInferenceContext(
        model_config=config,
        inference_config=InferenceConfig(
            max_sequence_length=2048,
            max_requests=8,
            max_tokens=2048,
            block_size_tokens=256,
            buffer_size_gb=0.2,
            unified_memory_level=0,
            num_cuda_graphs=4 if cuda_graphs else None,
            cuda_graph_max_tokens=256,
            materialize_only_last_token_logits=False,
            use_flashinfer_fused_rope=False,
        ),
    )
    controller = TextGenerationController(
        inference_wrapped_model=GPTInferenceWrapper(model, context),
        tokenizer=types.SimpleNamespace(
            vocab_size=VOCAB_SIZE, detokenize=lambda tokens, **kwargs: str(tokens[0])
        ),
    )
    delete_cuda_graphs()
    engine = DynamicInferenceEngine(controller, context)

    def add(request_id):
        engine._add_request(
            DynamicInferenceRequest(
                request_id=request_id,
                prompt_tokens=prompts[request_id],
                sampling_params=SamplingParams(
                    num_tokens_to_generate=GENERATE,
                    termination_id=-1,
                    top_k=1,
                    return_log_probs=True,
                    top_n_logprobs=2,
                ),
            )
        )

    finished, graphed_steps = {}, []

    def step():
        for request in engine.step_modern()["finished_requests"]:
            finished[request.request_id] = request
        graphed_steps.append(context.using_cuda_graph_this_step())

    for request_id in range(len(FIRST_PROMPTS)):
        add(request_id)
    for _ in range(2):  # Prefill, then decode.
        step()
    add(len(FIRST_PROMPTS))  # Prefilled in a step that also decodes the first two.
    while engine.has_unfinished_requests():
        step()
    delete_cuda_graphs()
    assert sorted(finished) == list(range(len(prompts)))
    # Decode steps replay CUDA graphs; the long prefills run eagerly.
    assert any(graphed_steps) == cuda_graphs and not all(graphed_steps)
    return finished


@pytest.mark.parametrize(
    "cuda_graphs, sequence_parallel",
    [pytest.param(False, True, id="eager"), pytest.param(True, True, id="cuda_graphs")],
)
def test_tensor_parallel_matches_one_rank(tmp_path_dist_ckpt, cuda_graphs, sequence_parallel):
    """TP2 (with or without sequence parallelism) reproduces TP1 tokens and log-probs."""
    pytest.importorskip("cutlass.cute")

    generator = torch.Generator().manual_seed(7)
    prompts = [
        torch.randint(VOCAB_SIZE, (length,), generator=generator)
        for length in (*FIRST_PROMPTS, LATE_PROMPT)
    ]
    served = {}
    Utils.initialize_model_parallel(1, 1)  # The checkpoint directory syncs ranks.
    name = f"dsa_gqa_tp_{cuda_graphs}_{sequence_parallel}"
    with TempNamedDir(tmp_path_dist_ckpt / name) as checkpoint:
        for tp in (1, 2):
            Utils.initialize_model_parallel(tp, 1)
            model_parallel_cuda_manual_seed(
                123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
            )
            torch.manual_seed(123)
            config = _config(tp, cuda_graphs, sequence_parallel)
            model = _model(config)
            if tp == 1:
                dist_checkpointing.save(model.sharded_state_dict(), checkpoint)
            else:
                # One KV group, replicated on both ranks; the TP2 linear_qkv shards hold
                # TP1's rows, and the indexer is duplicated on both ranks.
                assert model.decoder.layers[0].self_attention.num_query_groups_per_partition == 1
                model.load_state_dict(
                    dist_checkpointing.load(model.sharded_state_dict(), checkpoint)
                )
            served[tp] = _serve(model, config, prompts, cuda_graphs)
            del model
    Utils.destroy_model_parallel()

    mean_budget, max_budget = 1e-2, 0.1
    prompt_deltas, generated_deltas = [], []
    for request_id, reference in served[1].items():
        actual = served[2][request_id]
        prompt_deltas.append(
            (torch.tensor(actual.prompt_log_probs) - torch.tensor(reference.prompt_log_probs)).abs()
        )
        assert len(actual.generated_tokens) == len(reference.generated_tokens) == GENERATE
        for step, (expected, token) in enumerate(
            zip(reference.generated_tokens, actual.generated_tokens)
        ):
            if token != expected:
                # Only a near-tie between TP1's two best tokens may flip; later tokens then
                # follow a different history.
                top = sorted(reference.generated_top_n_logprobs[step].values(), reverse=True)
                assert top[0] - top[1] < 2 * max_budget, (
                    f"Request {request_id} step {step}: TP2 picked {token}, TP1 {expected} with "
                    f"top-2 log-probs {top}."
                )
                break
            generated_deltas.append(
                abs(actual.generated_log_probs[step] - reference.generated_log_probs[step])
            )
    prompt_deltas = torch.cat(prompt_deltas)
    generated_deltas = torch.tensor(generated_deltas)
    print(
        f"TP2 vs TP1 (graphs={cuda_graphs}, "
        f"sp={sequence_parallel}): prompt "
        f"mean {prompt_deltas.mean():.2e} max {prompt_deltas.max():.2e}; generated "
        f"{generated_deltas.numel()} tokens, max {generated_deltas.max():.2e}"
    )
    assert prompt_deltas.mean() < mean_budget and prompt_deltas.max() < max_budget
    assert generated_deltas.numel() >= len(served[1])
    assert generated_deltas.max() < max_budget


def test_training_rejects_sequence_parallelism():
    """Only dynamic-batching inference gathers the indexer's sequence shards; the training
    forward fails clearly instead of routing on a shard."""
    Utils.initialize_model_parallel(2, 1)
    model_parallel_cuda_manual_seed(123)
    model = _model(_config(tp=2, cuda_graphs=False, sequence_parallel=True)).train()
    tokens = torch.randint(VOCAB_SIZE, (1, 64), device="cuda")
    with pytest.raises(NotImplementedError, match="dynamic-batching inference only"):
        model(
            input_ids=tokens,
            position_ids=torch.arange(64, device="cuda").unsqueeze(0),
            attention_mask=None,
        )
    Utils.destroy_model_parallel()
