# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Source-backed GLM53 MTP composite/contract tests, not official-HF MTP parity.

HF93c8 supplies component arithmetic. Pinned independent GLM vLLM/SGLang
sources establish embedding-first, DE, no-inner-mHC and one final norm (see
glm53_native_reference.NativeMTP). CP1 training successor semantics are explicit
Torch copy/zero-fill operations, not a target roll/loss helper used as oracle.
All production widths, full288E inventory and vocabulary154880 are retained.

Attention core. The two composite tests build the reference with
attention_contract='absorbed': only the MTP attention core follows production's
absorbed-MLA contract (glm53_native_reference.absorbed_attention_core).
Embedding, norms, eh_proj, the q_a/q_b/kv_a/o_proj attention projections, the
indexer, router, all288 experts, decoder block and head remain HF93c8
arithmetic. kv_b_proj is not applied as the HF K/V expansion; its W_UK/W_UV
halves are consumed inside the absorbed core. Why: a pre-registered causal
diagnostic (eight workers agree, all eleven validity gates passed) found, for
the single-stream [none] case, every stage up to q, the latent and kv
bit-identical and the attention core as the first divergence. There, with the
absorbed contract, the reference reproduces production's MTP FFN input bit for
bit and passes every 0.999 bar (minimum gradient cosine 0.99996); with the HF
eager core 13.6% of that input differs, about ten SwiGLU clamp crossings move
and three gradient links fall below the bar (expert82 up 0.99805, minimum
0.9911). Score dtype and reassociation are both required. The diagnostic did
not cover the cudnn backend or the real HybridModel MTP case. The HF eager
attention stays under direct test in
test_glm53_mtp_attention_layer_matches_hf_eager. The cudnn backend (FlashMLA
forward, cuDNN backward) differs further from the Torch absorbed contract and
is not bitwise to it; its margin is whatever the run reports.

Comparison criteria of the real HybridModel case. The decoder output feeding
MTP differs from the reference only at rounding level, but that difference
moves SwiGLU clamp decisions in the MTP experts' std-0.5 coverage channels, and
the backward amplifies each moved decision into every upstream MTP gradient
(fed production's own decoder output, the reference passes the embedding
link). The case therefore replays the target's clamp decisions in the decoder
FFN and both MTP expert paths, and compares the decoder's shape-[1] mHC scale
gradients through their contribution tensors; see the hybrid module's
comparison criteria. All other links keep the 0.999 bars.
"""

import os

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_glm53_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    MultiTokenPredictionBlock,
)
from tests.unit_tests.models.glm53_native_reference import (
    HIDDEN,
    VOCAB,
    BufferContract,
    ClampReplay,
    ComparisonLedger,
    HostParameters,
    NativeCompositeModel,
    NativeMTPHarness,
    ParameterLink,
    assert_all_gradients,
    assert_mhc_decomposition,
    assert_similarity,
    balanced_router_weight,
    copy_parameters,
    linked_view,
    record_details,
    successor_tokens,
)
from tests.unit_tests.models.test_glm53_hybrid_native_parity import (
    BLOCK_MHC_SITES,
    assert_clamp_coverage,
    assert_sparse_observations,
    bind_moe_replay,
    block_links,
    capture_stream_wire,
    dsa_links,
    initialize_fixture,
    make_config,
    moe_links,
    observe_clamps,
    observe_dsa,
    observe_expert_dispatch,
    observe_kda_convolution,
    observe_kda_core_precision,
    observe_mhc_projections,
    process_groups,
    record_router,
    release_cuda,
)
from tests.unit_tests.test_utilities import Utils


class RealMTPHarness(nn.Module):
    """Test assembly of actual public embedding/MTP/head modules; no fake branch."""

    def __init__(self, config):
        super().__init__()
        groups = process_groups()
        self.embedding = LanguageModelEmbedding(
            config=config,
            vocab_size=VOCAB,
            max_sequence_length=130,
            position_embedding_type='none',
            tp_group=groups.tp,
            pg_collection=groups,
        )
        spec = hybrid_glm53_stack_spec(config).submodules
        self.mtp = MultiTokenPredictionBlock(
            config=config,
            spec=spec.mtp_block_spec,
            mtp_layer_pattern='DE',
            mtp_num_depths=1,
            hybrid_submodules=spec,
            pg_collection=groups,
        )
        self.output_layer = tensor_parallel.ColumnParallelLinear(
            HIDDEN,
            VOCAB,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=True,
            tp_group=groups.tp,
            pg_collection=groups,
        )

    def forward(self, hidden, tokens):
        positions = torch.arange(tokens.size(1), device=tokens.device).expand_as(tokens)
        output = self.mtp(
            hidden_states=hidden,
            input_ids=tokens,
            position_ids=positions,
            attention_mask=None,
            embedding=self.embedding,
        )
        main, predicted = output.chunk(2, 0)
        main_logits, bias = self.output_layer(main)
        assert bias is None
        predicted_logits, bias = self.output_layer(predicted)
        assert bias is None
        return output, main_logits, predicted_logits


def mtp_links():
    prefix = 'mtp.layers.0'
    attention = f'{prefix}.mtp_model_layer.layers.0'
    ffn = f'{prefix}.mtp_model_layer.layers.1'
    return [
        ParameterLink('embedding.weight', 'embedding.word_embeddings.weight'),
        ParameterLink('output_layer.weight', 'output_layer.weight'),
        ParameterLink('mtp.enorm.weight', f'{prefix}.enorm.weight'),
        ParameterLink('mtp.hnorm.weight', f'{prefix}.hnorm.weight'),
        ParameterLink('mtp.eh_proj.weight', f'{prefix}.eh_proj.weight'),
        ParameterLink('mtp.input_layernorm.weight', f'{attention}.input_layernorm.weight'),
        ParameterLink('mtp.post_attention_layernorm.weight', f'{ffn}.pre_mlp_layernorm.weight'),
        ParameterLink('mtp.final_norm.weight', f'{prefix}.final_layernorm.weight'),
        *dsa_links('mtp.self_attn', f'{attention}.self_attention'),
        *moe_links('mtp.mlp', f'{ffn}.mlp'),
    ]


def model_links():
    return mtp_links() + [
        ParameterLink(
            f'decoder.{link.native}', f'decoder.{link.actual}', link.native_slice, link.actual_slice
        )
        for link in block_links()
    ]


def actual_router(model):
    return model.mtp.layers[0].mtp_model_layer.layers[1].mlp.router


def mtp_buffer_contract(actual, native):
    return BufferContract(
        actual,
        native,
        [
            (
                'mtp.layers.0.mtp_model_layer.layers.1.mlp.router.expert_bias',
                'mtp.mlp.e_score_correction_bias',
            )
        ],
        ['mtp.layers.0.mtp_model_layer.layers.1.mlp.router.local_tokens_per_expert'],
    )


def assert_single_stream_inventory(model):
    layer = model.mtp.layers[0]
    assert not layer.mhc_enabled
    assert not layer.mtp_model_layer.config.enable_hyper_connections
    assert layer.eh_proj.weight.shape == (HIDDEN, 2 * HIDDEN)
    assert layer.e_proj is None and layer.h_proj is None
    assert not any(
        'hyper_connection' in name or 'hc_head_' in name for name, _ in model.mtp.named_parameters()
    )


@torch.no_grad()
def set_router_fixture(model, native_moe, prefix_input):
    router = actual_router(model)
    router.expert_bias.data = router.expert_bias.data.float()
    bias = torch.linspace(-0.02, 0.02, 288, device='cuda')
    router.expert_bias.copy_(bias)
    native_moe.e_score_correction_bias.copy_(bias)
    router.weight.copy_(balanced_router_weight(prefix_input))
    return bias


def tokens_fixture():
    return (torch.arange(4 * 130, device='cuda').reshape(4, 130) + 100).long()


@pytest.fixture(autouse=True)
def _model_parallel(request, record_property):
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(531531)
    torch.manual_seed(531531)
    old_tracker = MTPLossLoggingHelper.tracker
    old_scale = MTPLossAutoScaler.main_loss_backward_scale
    MTPLossLoggingHelper.tracker = {}
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0, device='cuda'))
    with ComparisonLedger(request.node.nodeid, int(os.environ.get('RANK', '0'))) as ledger:
        try:
            yield
        finally:
            ledger.publish(record_property)
            MTPLossLoggingHelper.tracker = old_tracker
            MTPLossAutoScaler.set_loss_scale(old_scale)
            Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize('backend', ['none', 'cudnn'])
def test_glm53_single_stream_mtp_native_output_input_and_all_gradients(backend, monkeypatch):
    """Depth1 DE and shared head, every288 expert routed, implementations sequential.

    Both reference harnesses (router fixture and comparison) use the absorbed
    attention core and otherwise HF93c8 arithmetic; see the module docstring
    for the diagnostic evidence. [none] is the case that evidence covers;
    [cudnn] is FlashMLA/cuDNN and not bitwise to the Torch absorbed core.
    """
    config = make_config(backend=backend, sparse=True, mtp=True)
    real = RealMTPHarness(config).cuda()
    convert_module_to_dtype_except_fp32_marked(real, torch.bfloat16)
    initialize_fixture(real)
    assert_single_stream_inventory(real)
    native = NativeMTPHarness(torch.device('cuda'), attention_contract='absorbed')
    links = mtp_links()
    copy_parameters(real, native, links)
    tokens = tokens_fixture()
    hidden = torch.randn(130, 4, HIDDEN, device='cuda', dtype=torch.bfloat16)
    with torch.no_grad():
        set_router_fixture(real, native.mtp.mlp, native.ffn_input(hidden, tokens))
    buffers = mtp_buffer_contract(real, native)
    del native
    release_cuda()
    actual_input = hidden.clone().requires_grad_()
    router_records, norm_calls = [], []
    router_hook = record_router(actual_router(real), router_records)
    norm_hook = real.mtp.layers[0].final_layernorm.register_forward_hook(
        lambda *_: norm_calls.append(1)
    )
    try:
        with observe_dsa(monkeypatch) as records, observe_clamps(real) as clamp_records:
            output = real(actual_input, tokens)
            gradients = tuple(torch.randn_like(value) for value in output)
            torch.autograd.backward(output, gradients)
    finally:
        router_hook.remove()
        norm_hook.remove()
    assert norm_calls == [1], 'single-stream MTP must postprocess exactly once'
    assert output[0].shape == (260, 4, HIDDEN)
    output_gold = tuple(value.detach().cpu() for value in output)
    input_gold = actual_input.grad.detach().cpu()
    buffers.assert_actual_unchanged(real, router_records)
    gold = HostParameters(real)
    del real, output, actual_input
    release_cuda()
    native = NativeMTPHarness(torch.device('cuda'), attention_contract='absorbed')
    copy_parameters(gold, native, links)
    buffers.copy_native(gold, native)
    native_input = hidden.clone().requires_grad_()
    expected = native(native_input, tokens)
    for name, actual, reference in zip(
        ('hidden', 'main logits', 'MTP logits'), output_gold, expected
    ):
        assert_similarity(actual, reference, f'MTP {name}')
    torch.autograd.backward(expected, gradients)
    assert_similarity(input_gold, native_input.grad, 'MTP incoming hidden gradient')
    assert_sparse_observations(
        records, router_records, native.mtp.self_attn, native.mtp.mlp, backend
    )
    assert_clamp_coverage(clamp_records, native)
    buffers.assert_native_unchanged(native)
    assert_all_gradients(gold, native, links)
    record_details('objective', 'paired main+MTP hidden and both shared-head logit random VJPs')


def cross_entropy(logits, labels):
    sequence, batch = logits.shape[:2]
    return (
        F.cross_entropy(
            logits.float().reshape(-1, VOCAB), labels.T.contiguous().reshape(-1), reduction='none'
        )
        .reshape(sequence, batch)
        .T.contiguous()
    )


@pytest.mark.internal
def test_glm53_real_hybrid_mtp_contract_vjp_and_all_gradients(monkeypatch):
    """Real K-/DE forward/loss handoff from a legal distinct-stream wire input.

    This TP1 public intermediate-stage entry avoids unobservable first-site mHC
    mixing from identical expanded streams. It is not a PP2 communication test.
    The normal first-stage expansion boundary is independently checked in A.
    Objective: mainCE+0.1*validMTPCE plus a declared full-stream-wire random
    VJP, needed to observe residual mixing hidden by the final mean head. This
    multi-output Jacobian test is NOT ordinary optimizer-step training evidence.
    Both reference models use the absorbed MTP attention core (module docstring);
    the KDA+dense block and the rest of MTP stay HF93c8. This case runs cudnn
    (FlashMLA/cuDNN), which is not bitwise to the Torch absorbed core and which
    the diagnostic did not cover. Clamp decisions are replayed and the decoder
    mHC scales are decomposed (module docstring).
    """
    backend = 'cudnn'
    config = make_config(deterministic=True, backend=backend, sparse=True, mtp=True)
    real = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_glm53_stack_spec(config),
        vocab_size=VOCAB,
        max_sequence_length=130,
        hybrid_layer_pattern='K-/DE',
        pre_process=False,
        post_process=True,
        parallel_output=False,
        share_embeddings_and_output_weights=False,
        position_embedding_type='none',
        pg_collection=process_groups(),
    ).cuda()
    convert_module_to_dtype_except_fp32_marked(real, torch.bfloat16)
    initialize_fixture(real)
    assert_single_stream_inventory(real)
    links = model_links()
    native = NativeCompositeModel(torch.device('cuda'), wire=True, attention_contract='absorbed')
    copy_parameters(real, native, links)
    tokens = tokens_fixture()
    labels = tokens + 1
    positions = torch.arange(130, device='cuda').expand_as(tokens)
    hidden = torch.randn(130, 4, 4 * HIDDEN, device='cuda', dtype=torch.bfloat16)
    with torch.no_grad():
        set_router_fixture(real, native.mtp.mlp, native.ffn_input(tokens, hidden))
    buffers = mtp_buffer_contract(real, native)
    del native
    release_cuda()
    actual_input = hidden.clone().requires_grad_()
    real.set_input_tensor(actual_input)
    heads, mtp_outputs, norm_calls, router_records, wires, fc1 = [], [], [], [], [], {}

    def capture_head(_module, _inputs, output):
        assert output[1] is None
        heads.append(output[0].detach().cpu())

    hooks = [
        real.output_layer.register_forward_hook(capture_head),
        real.mtp.register_forward_hook(
            lambda _m, _i, output: mtp_outputs.append(output.detach().cpu())
        ),
        real.mtp.layers[0].final_layernorm.register_forward_hook(lambda *_: norm_calls.append(1)),
        record_router(actual_router(real), router_records),
        capture_stream_wire(real.decoder, wires),
    ]
    try:
        with (
            observe_dsa(monkeypatch) as records,
            observe_clamps(real, fc1) as clamp_records,
            observe_expert_dispatch(real) as dispatch,
            observe_kda_convolution(monkeypatch, True),
            observe_kda_core_precision(real, monkeypatch),
            observe_mhc_projections(real, monkeypatch) as projections,
        ):
            loss = real(
                input_ids=tokens,
                position_ids=positions,
                attention_mask=None,
                labels=labels,
                loss_mask=torch.ones_like(tokens),
            )
            assert len(wires) == 1
            wire_gradient = torch.randn_like(wires[0])
            objective = loss.mean()
            torch.autograd.backward(
                (objective, wires[0]), (torch.ones_like(objective), wire_gradient)
            )
    finally:
        for hook in hooks:
            hook.remove()
    assert len(heads) == 2 and len(mtp_outputs) == 1 and norm_calls == [1]
    input_gold, loss_gold = actual_input.grad.detach().cpu(), loss.detach().cpu()
    wire_gold = wires[0].detach().cpu()
    mtp_loss_gold = MTPLossLoggingHelper.tracker['loss_sums'][0].detach().cpu()
    projections = {
        name: {key: value.cpu() for key, value in entry.items()}
        for name, entry in projections.items()
    }
    buffers.assert_actual_unchanged(real, router_records)
    gold = HostParameters(real)
    wires.clear()
    del real, loss, actual_input, objective
    release_cuda()
    native = NativeCompositeModel(torch.device('cuda'), wire=True, attention_contract='absorbed')
    copy_parameters(gold, native, links)
    buffers.copy_native(gold, native)
    replay = ClampReplay()
    replay.bind(
        'decoder.layers.1.inner_layer.mlp',
        native.decoder.mlp,
        fc1.pop('decoder.layers.1.inner_layer.mlp.linear_fc1'),
    )
    bind_moe_replay(
        replay,
        'mtp.layers.0.mtp_model_layer.layers.1.mlp',
        native.mtp.mlp,
        fc1,
        dispatch,
        router_records,
    )
    assert not fc1 and not dispatch, (sorted(fc1), sorted(dispatch))
    native_input = hidden.clone().requires_grad_()
    hidden_output, main_logits, mtp_logits = native(tokens, native_input)
    replay.assert_consumed()
    assert_similarity(mtp_outputs[0], hidden_output, 'real HybridModel main+MTP hidden handoff')
    # The actual training path invokes MTP head first, then main head.
    assert_similarity(heads[0], mtp_logits, 'real HybridModel MTP head invocation0')
    assert_similarity(heads[1], main_logits, 'real HybridModel main head invocation1')
    main_loss = cross_entropy(main_logits, labels)
    assert_similarity(loss_gold, main_loss, 'real HybridModel main CE')
    assert_similarity(wire_gold, native.decoder.last_streams, 'real HybridModel full stream wire')
    mtp_loss = cross_entropy(mtp_logits, successor_tokens(labels))
    valid = torch.ones_like(mtp_loss)
    valid[:, -1] = 0
    mtp_mean = (mtp_loss * valid).sum() / valid.sum()
    assert_similarity(mtp_loss_gold, mtp_mean, 'real HybridModel valid MTP CE')
    objective = main_loss.mean() + 0.1 * mtp_mean
    torch.autograd.backward(
        (objective, native.decoder.last_streams), (torch.ones_like(objective), wire_gradient)
    )
    record_details(
        'objective',
        'mainCE + 0.1*validMTPCE + declared complete-stream-wire random VJP; not an ordinary training step',
    )
    assert_similarity(input_gold, native_input.grad, 'real HybridModel wire gradient')
    assert_sparse_observations(
        records, router_records, native.mtp.self_attn, native.mtp.mlp, backend
    )
    assert_clamp_coverage(clamp_records, native)
    buffers.assert_native_unchanged(native)
    decomposed = assert_mhc_decomposition(
        gold,
        native,
        projections,
        [(f'decoder.{actual}', f'decoder.{name}') for actual, name in BLOCK_MHC_SITES],
    )
    assert_all_gradients(gold, native, links, decomposed=decomposed)


@pytest.mark.internal
@pytest.mark.parametrize('backend', ['none', 'cudnn'])
def test_glm53_mtp_attention_layer_matches_hf_eager(backend, monkeypatch):
    """Production MTP attention layer versus the HF eager reference attention.

    Local complement to the composite tests, whose reference swaps in the
    absorbed core: the default 'hf_eager' NativeAttention (expanded BF16 K/V,
    BF16 QK logits, FP32 softmax, BF16 P x V) is compared with the production
    layer on one input, the production attention-layer input captured from a
    no-grad forward of the full harness. Both sides get identical fresh leaf
    clones and one deterministic random cotangent. Output, input gradient and
    every attention parameter gradient are held to the 0.999 bars; the frozen
    indexer has no gradient on either side. The core rounding difference is
    observed here before downstream SwiGLU clamp crossings can amplify it.
    """
    attention_name = 'mtp.layers.0.mtp_model_layer.layers.0.self_attention'
    config = make_config(backend=backend, sparse=True, mtp=True)
    real = RealMTPHarness(config).cuda()
    convert_module_to_dtype_except_fp32_marked(real, torch.bfloat16)
    initialize_fixture(real)
    assert_single_stream_inventory(real)
    native = NativeMTPHarness(torch.device('cuda'))
    copy_parameters(real, native, mtp_links())
    # Only the reference attention is needed; free the rest of the copy now.
    native_attention = native.mtp.self_attn
    assert native_attention.core_contract == 'hf_eager'
    del native
    release_cuda()
    real_attention = real.mtp.layers[0].mtp_model_layer.layers[0].self_attention
    calls = []

    def detached(value):
        return value.detach().clone() if torch.is_tensor(value) else value

    def capture_call(_module, args, kwargs):
        calls.append(
            (tuple(detached(value) for value in args), {k: detached(v) for k, v in kwargs.items()})
        )

    hook = real_attention.register_forward_pre_hook(capture_call, with_kwargs=True)
    try:
        with torch.no_grad():
            real(torch.randn(130, 4, HIDDEN, device='cuda', dtype=torch.bfloat16), tokens_fixture())
    finally:
        hook.remove()
    assert len(calls) == 1, 'depth-1 MTP must call its attention layer exactly once'
    args, kwargs = calls.pop()
    release_cuda()
    hidden_positional = bool(args)
    if hidden_positional:
        layer_input, args = args[0], args[1:]
    else:
        layer_input = kwargs.pop('hidden_states')
    assert layer_input.shape == (130, 4, HIDDEN) and layer_input.dtype == torch.bfloat16
    captured_kwargs = sorted(kwargs)
    actual_input = layer_input.clone().requires_grad_()
    with observe_dsa(monkeypatch) as records:
        output, bias = real_attention(actual_input, *args, **kwargs)
    assert bias is None
    assert output.shape == layer_input.shape
    cotangent = torch.randn(
        output.shape,
        generator=torch.Generator(device='cuda').manual_seed(531531),
        device='cuda',
        dtype=output.dtype,
    )
    output.backward(cotangent)
    links = dsa_links('mtp.self_attn', attention_name)
    actual_parameters = dict(real_attention.named_parameters(prefix=attention_name))
    assert set(actual_parameters) == {link.actual for link in links}
    output_gold = output.detach().cpu()
    input_gold = actual_input.grad.detach().cpu()
    gradient_gold = {}
    for link in links:
        parameter = actual_parameters[link.actual]
        assert parameter.requires_grad == ('.indexer.' not in link.native), link
        if parameter.requires_grad:
            assert parameter.grad is not None, link
            gradient = linked_view(parameter.grad, link.actual_slice)
            gradient_gold[link.actual] = gradient.detach().cpu()
        else:
            assert parameter.grad is None, link
            gradient_gold[link.actual] = None
    del real, real_attention, actual_parameters, output, actual_input, args, kwargs
    del parameter, gradient
    release_cuda()
    record_details(
        'attention_layer',
        {
            'backend': backend,
            'production_module': attention_name,
            'reference_core_contract': native_attention.core_contract,
            'input': 'production attention-layer input captured in a no-grad harness forward',
            'hidden_states_argument': 'positional' if hidden_positional else 'keyword',
            'captured_kwargs': captured_kwargs,
            'fused_returned_non_none': records['fused'],
            'cotangent': 'torch.randn with a dedicated CUDA generator, seed 531531',
            'frozen_links_without_gradient': [
                link.native for link in links if gradient_gold[link.actual] is None
            ],
        },
    )
    native_input = layer_input.clone().requires_grad_()
    expected = native_attention(native_input)
    expected.backward(cotangent)
    assert_similarity(output_gold, expected, 'MTP attention layer output')
    assert_similarity(input_gold, native_input.grad, 'MTP attention layer input gradient')
    assert len(records['indices']) == 1
    assert records['fused'] == ([True] if backend == 'cudnn' else [])
    torch.testing.assert_close(
        records['indices'][0].sort(-1).values,
        native_attention.last_indices.cpu().sort(-1).values,
        rtol=0,
        atol=0,
    )
    native_parameters = dict(native_attention.named_parameters(prefix='mtp.self_attn'))
    assert set(native_parameters) == {link.native for link in links}
    for link in links:
        gradient = native_parameters[link.native].grad
        if gradient_gold[link.actual] is None:
            assert gradient is None, link
        else:
            assert gradient is not None, link
            assert_similarity(
                gradient_gold[link.actual],
                linked_view(gradient, link.native_slice),
                f'{link.actual} <- {link.native}',
            )
    record_details('objective', 'MTP attention-layer random VJP on a captured production input')
