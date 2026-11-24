# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Disaggregated DSA-over-GQA serving matches aggregated serving.

Even ranks prefill and hand off; odd ranks import the KV and DSA index-key pages and decode. The
imported caches must equal the prefill's bytes. A second request shares the first two blocks, so
the decode takes them from its prefix cache and imports only the suffix.

Sparse decode attention sums the selected keys in the order the top-k kernel returns them, which
varies between runs, so aggregated serving is itself not bit-reproducible. Only the FlashInfer
backend in batch-invariant mode sorts the selection; there disaggregated serving must match
aggregated serving exactly. Elsewhere the outputs must agree within the run-to-run noise of
aggregated serving, which the test measures and reports.
"""

import types
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.disaggregation.engine import StateHandoffDynamicInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant import dsa_gqa_flashinfer
from megatron.core.transformer.experimental_attention_variant.dsa_layer_specs import dsa_stack_spec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tuning import set_deterministic_mode
from tests.unit_tests.inference.engines.disagg_test_utils import (
    FLASH_ATTENTION_VERSION,
    admit_import,
    assert_import_equal,
    assert_released,
    complete_transfer,
    exchange,
    run_to_completion,
    sampling,
    snapshot_source,
)
from tests.unit_tests.test_utilities import Utils

VOCAB_SIZE, BLOCK, TOPK, GENERATE = 1024, 256, 512, 8
# Longer than the top-k, ending in a partial block; the second prompt shares two full blocks.
FIRST, SECOND = 600, 700
SHARED = 2 * BLOCK
# Largest log-prob difference allowed outside batch-invariant mode. Aggregated serving differs
# from itself (repeat run, other GPU) by up to ~4.4e-3 (BF16, Triton and FlashInfer); the budget
# is ~5x that. NVFP4 ties more often at the top-k boundary, so it gets a wider budget.
LOGPROB_BUDGET = {"bf16": 2e-2, "nvfp4": 5e-2}

pytestmark = [
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
        reason="Dynamic DSA-over-GQA inference requires SM100/SM103",
    ),
    pytest.mark.skipif(Utils.world_size < 2, reason="Needs a prefill and a decode rank"),
]


def _config(batch_invariant=False):
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
        inference_rng_tracker=True,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_indexer_n_heads=1,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=TOPK,
        dsa_indexer_loss_coeff=0.1,
        batch_invariant_mode=batch_invariant,
        # Batch-invariant mode pins the dense attention kernel to FlashAttention 3/4.
        **(
            dict(
                attention_backend=AttnBackend.flash, flash_attention_version=FLASH_ATTENTION_VERSION
            )
            if batch_invariant
            else {}
        ),
    )
    config.rope_type = "rope"
    config.rotary_base = 10000
    config.rotary_percent = 1.0
    return config


def _model(config):
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
        dist.broadcast(param.data, src=0)  # Identical weights on every rank.
    return model.eval()


@contextmanager
def _engine(model, config, dsa_backend, index_key_dtype, role=None):
    context = DynamicInferenceContext(
        model_config=config,
        inference_config=InferenceConfig(
            max_sequence_length=2048,
            max_requests=4,
            max_tokens=2048,
            block_size_tokens=BLOCK,
            buffer_size_gb=0.2,
            unified_memory_level=0,
            materialize_only_last_token_logits=False,
            use_flashinfer_fused_rope=False,
            enable_prefix_caching=True,
            kv_transport_backend="nccl",
            disaggregation_role=role,
            dsa_backend=dsa_backend,
            dsa_indexer_key_dtype=index_key_dtype,
        ),
    )
    controller = TextGenerationController(
        inference_wrapped_model=GPTInferenceWrapper(model, context),
        tokenizer=types.SimpleNamespace(
            vocab_size=VOCAB_SIZE, detokenize=lambda tokens, **kwargs: str(tokens[0])
        ),
    )
    delete_cuda_graphs()
    engine = StateHandoffDynamicInferenceEngine(controller, context)
    if role is not None:
        engine.setup_kv_transfer(role)
    try:
        yield engine
    finally:
        for socket in getattr(engine, "zmq_sockets", []):
            socket.close(linger=0)
        delete_cuda_graphs()


def _params(count=GENERATE, top_n=0, **changes):
    # Handoffs do not carry top-n log-probs; only the aggregated reference records them.
    return sampling(
        count, return_log_probs=True, skip_prompt_log_probs=True, top_n_logprobs=top_n, **changes
    )


def _result(request):
    tops = request.generated_top_n_logprobs or []
    return (
        list(request.generated_tokens),
        list(request.generated_log_probs),
        [sorted(top.values(), reverse=True) for top in tops],
    )


def _serve_aggregated(model, config, prompts, dsa_backend, index_key_dtype):
    with _engine(model, config, dsa_backend, index_key_dtype) as engine:
        return {
            request_id: _result(
                run_to_completion(engine, engine.add_request(request_id, tokens, _params(top_n=2)))
            )
            for request_id, tokens in prompts.items()
        }


def _divergence(actual, reference, budget):
    """Max log-prob delta over the steps both runs agree on, and the first differing step.

    A different token is allowed only where the reference's two best log-probs are within the
    budget of each other (a near tie); later steps then follow another history.
    """
    deltas, flipped = [0.0], None
    for step, (token, expected) in enumerate(zip(actual[0], reference[0])):
        if token != expected:
            top = reference[2][step]
            assert (
                top[0] - top[1] < 2 * budget
            ), f"step {step}: {token} vs {expected}, reference top-2 log-probs {top}"
            flipped = step
            break
        deltas.append(abs(actual[1][step] - reference[1][step]))
    return max(deltas), flipped


@contextmanager
def _batch_invariant(enabled):
    """Batch-invariant kernels; a hybrid model also needs deterministic kernel selection."""
    if not enabled:
        yield
        return
    set_deterministic_mode(True)
    bik.enable_batch_invariant_mode(backend="te_native", collective="ordered")
    try:
        yield
    finally:
        bik.disable_batch_invariant_mode()
        set_deterministic_mode(None)


@pytest.fixture
def transport_world():
    Utils.initialize_model_parallel(1, 1)
    assert dist.get_world_size() % 2 == 0
    control = dist.new_group(backend="gloo")
    yield control
    dist.destroy_process_group(control)
    Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "dsa_backend, index_key_dtype, batch_invariant",
    [
        pytest.param("flashinfer", "bf16", True, id="flashinfer-bf16-batch_invariant"),
        pytest.param("flashinfer", "bf16", False, id="flashinfer-bf16"),
        pytest.param("flashinfer", "nvfp4", False, id="flashinfer-nvfp4"),
        pytest.param("triton", "bf16", False, id="triton-bf16"),
    ],
)
@torch.inference_mode()
def test_disaggregated_dsa_matches_aggregated(
    transport_world, dsa_backend, index_key_dtype, batch_invariant
):
    pytest.importorskip("cutlass.cute")
    if dsa_backend == "triton":
        from triton.experimental.gluon.language.nvidia.hopper import tma

        if not hasattr(tma, "async_load"):
            pytest.skip("The Triton DSA kernels need Gluon's tma.async_load")
    if dsa_backend == "flashinfer" and not dsa_gqa_flashinfer.HAVE_FLASHINFER_DSA:
        pytest.skip("FlashInfer's DSA kernels are not installed")
    if index_key_dtype == "nvfp4" and not dsa_gqa_flashinfer.HAVE_FLASHINFER_DSA_NVFP4:
        pytest.skip("FlashInfer's NVFP4 indexer kernels are not installed")

    with _batch_invariant(batch_invariant):
        model_parallel_cuda_manual_seed(
            123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )
        torch.manual_seed(123)
        config = _config(batch_invariant)
        model = _model(config)
        generator = torch.Generator().manual_seed(11)
        first = torch.randint(VOCAB_SIZE, (FIRST,), generator=generator).tolist()
        second = (
            first[:SHARED]
            + torch.randint(VOCAB_SIZE, (SECOND - SHARED,), generator=generator).tolist()
        )
        prompts = {101: first, 103: second}
        budget = LOGPROB_BUDGET[index_key_dtype]
        label = f"{dsa_backend}/{index_key_dtype}/batch_invariant={batch_invariant}"

        # Aggregated noise floor: two runs here, and the peer rank's run.
        expected = _serve_aggregated(model, config, prompts, dsa_backend, index_key_dtype)
        repeat = _serve_aggregated(model, config, prompts, dsa_backend, index_key_dtype)
        peer = exchange(expected, transport_world)
        for request_id in prompts:
            for name, other in (("repeat", repeat), ("peer", peer)):
                delta, flipped = _divergence(other[request_id], expected[request_id], budget)
                print(
                    f"DSA_AGG_FLOOR {label} request={request_id} vs={name} "
                    f"max_logprob_delta={delta:.3e} first_flip={flipped}",
                    flush=True,
                )
        if batch_invariant:
            assert repeat == expected and peer == expected

        source = dist.get_rank() % 2 == 0
        role = "prefill" if source else "decode"
        with _engine(model, config, dsa_backend, index_key_dtype, role) as engine:
            assert engine._dsa_local_meta["index_key_dtype"] == index_key_dtype
            for request_id, tokens in prompts.items():
                metadata = state = None
                if source:
                    request = run_to_completion(
                        engine,
                        engine.add_request(request_id, tokens, _params(1, do_kv_handoff=True)),
                    )
                    metadata, state = snapshot_source(engine, request, request_id)
                transferred = exchange((metadata, state) if source else None, transport_world)
                if not source:
                    metadata, state = transferred
                pending, future = complete_transfer(
                    engine, metadata, tokens, _params(), transport_world, request_id
                )
                if source:
                    engine._poll_pending_kv_pushes()
                    engine.release_handoff_blocks(request_id)
                else:
                    assert pending.cached_prefix_block_count == (2 if request_id == 103 else 0)
                    # The imported KV and index-key pages equal the prefill's bytes.
                    assert_import_equal(engine, pending, state, len(tokens))
                    admit_import(engine)
                    actual = _result(run_to_completion(engine, future))
                    delta, flipped = _divergence(actual, expected[request_id], budget)
                    print(
                        f"DSA_DISAGG {label} request={request_id} "
                        f"cached_blocks={pending.cached_prefix_block_count} tokens={actual[0]} "
                        f"expected={expected[request_id][0]} max_logprob_delta={delta:.3e} "
                        f"first_flip={flipped}",
                        flush=True,
                    )
                    if batch_invariant:
                        # Tokens and log-probs; handoffs carry no top-n log-probs.
                        assert actual[:2] == expected[request_id][:2]
                    else:
                        assert delta < budget
                dist.barrier(group=transport_world)
            assert_released(engine)
