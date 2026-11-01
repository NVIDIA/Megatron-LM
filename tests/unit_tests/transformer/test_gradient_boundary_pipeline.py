# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Frozen-prefix gradient boundaries across pipeline stages."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import NO_TRAINABLE_LAYER, reset_gradient_boundary_cache
from tests.unit_tests.test_utilities import Utils

NUM_LAYERS, VOCAB, SEQ, MBS, MICROBATCHES = 4, 128, 16, 2, 3
# Attention and MLP layers alternate, so every pipeline split keeps both kinds.
HYBRID_LAYERS = "*-" * (NUM_LAYERS // 2)


@pytest.fixture
def pipeline_groups(pipeline_size):
    if Utils.world_size < pipeline_size or Utils.world_size % pipeline_size:
        pytest.skip(f"{pipeline_size} pipeline stages need a multiple of {pipeline_size} ranks")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=pipeline_size
    )
    model_parallel_cuda_manual_seed(1234)
    yield
    Utils.destroy_model_parallel()


def _build_model(kind: str, pipeline_size: int, recompute: bool):
    extra = (
        dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)
        if recompute
        else {}
    )
    config = TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        pipeline_model_parallel_size=pipeline_size,
        pipeline_dtype=torch.float32,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        gradient_accumulation_fusion=False,
        **extra,
    )
    stage_flags = dict(
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
    )
    if kind == "hybrid":
        # HybridStack full recompute goes through checkpointed_forward without a layer
        # offset, and its layers carry global layer numbers on every stage.
        per_stage = NUM_LAYERS // pipeline_size
        pattern = "|".join(
            HYBRID_LAYERS[start : start + per_stage] for start in range(0, NUM_LAYERS, per_stage)
        )
        return HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=VOCAB,
            max_sequence_length=SEQ,
            hybrid_layer_pattern=pattern,
            **stage_flags,
        ).cuda()
    return GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        **stage_flags,
    ).cuda()


def _train_from(model, first_trainable: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in model.decoder.layers:
        if layer.layer_number - 1 >= first_trainable:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
    final_norm = getattr(model.decoder, "final_layernorm", None) or getattr(
        model.decoder, "final_norm", None
    )
    if model.post_process and final_norm is not None:
        for parameter in final_norm.parameters():
            parameter.requires_grad_(True)


def _run(model) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    batches = iter(
        [torch.randint(0, VOCAB, (MBS, SEQ), generator=generator) for _ in range(MICROBATCHES)]
    )

    def forward_step(data_iterator, model):
        tokens = next(data_iterator).cuda()
        positions = torch.arange(SEQ, device="cuda").expand_as(tokens)
        output = model(tokens, positions, None, labels=tokens)

        def loss_func(output_tensor):
            loss = output_tensor.float().mean()
            return loss, {"loss": loss.detach()}

        return output, loss_func

    get_forward_backward_func()(
        forward_step_func=forward_step,
        data_iterator=batches,
        model=[model],
        num_microbatches=MICROBATCHES,
        seq_length=SEQ,
        micro_batch_size=MBS,
        forward_only=False,
    )
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    for parameter in model.parameters():
        parameter.grad = None
    return grads


def _first(output):
    return output[0] if isinstance(output, tuple) else output


@pytest.mark.parametrize("kind", ["gpt", "hybrid"])
@pytest.mark.parametrize("recompute", [False, True], ids=["eager", "full-recompute"])
@pytest.mark.parametrize(
    "pipeline_size,first_trainable",
    [(2, 2), (2, 3), (4, 2), (4, 3), (2, NUM_LAYERS), (4, NUM_LAYERS)],
    ids=[
        "pp2-boundary-at-stage-start",
        "pp2-boundary-inside-stage",
        "pp4-one-frozen-middle-stage",
        "pp4-two-frozen-middle-stages",
        "pp2-all-layers-frozen",
        "pp4-all-layers-frozen",
    ],
)
def test_pipeline_frozen_prefix_matches_full_backward(
    pipeline_groups, pipeline_size, first_trainable, recompute, kind
):
    """Stages agree on the boundary, skip frozen work and send zero input gradients.

    Without the agreed boundary, a later stage would run its frozen layers under
    autograd because its received input requires grad. The reference keeps that
    behavior.
    """
    model = _build_model(kind, pipeline_size, recompute)
    _train_from(model, first_trainable)
    layers = list(model.decoder.layers)
    stage_is_frozen = layers[-1].layer_number - 1 < first_trainable
    frozen_sublayer_flags, stage_output_flags = [], []

    def record_sublayer(module, args, output):
        frozen_sublayer_flags.append(_first(output).requires_grad)

    def record_stage_output(module, args, output):
        stage_output_flags.append(_first(output).requires_grad)

    for layer in layers:
        if layer.layer_number - 1 < first_trainable:
            # Attention and MLP are the frozen layers' real work. Bias-dropout-add runs
            # under enable_grad by design, so the layer output itself is not checked.
            # Hybrid layers hold one of the two; the other is a parameter-free identity.
            for sublayer in (layer.self_attention, layer.mlp):
                if any(True for _ in sublayer.parameters()):
                    sublayer.register_forward_hook(record_sublayer)
    model.decoder.register_forward_hook(record_stage_output)

    grads = _run(model)
    # With every layer frozen only the final norm trains, and no stage needs autograd
    # through its layers.
    expected_boundary = first_trainable if first_trainable < NUM_LAYERS else NO_TRAINABLE_LAYER
    assert model.decoder._pipeline_gradient_boundary_layer == expected_boundary
    boundary_sublayer_flags = list(frozen_sublayer_flags)
    boundary_output_flags = list(stage_output_flags)
    frozen_sublayer_flags.clear()
    stage_output_flags.clear()

    reset_gradient_boundary_cache(model)
    model.decoder._pipeline_gradient_boundary_synced = True
    model.decoder._pipeline_gradient_boundary_layer = None
    model.decoder._gradient_boundary_layer_cache = {False: None, True: None}
    reference = _run(model)
    reference_sublayer_flags = list(frozen_sublayer_flags)

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert set(reference) == trainable
    assert set(grads) == trainable
    for name in trainable:
        torch.testing.assert_close(grads[name], reference[name], atol=1e-6, rtol=1e-5, msg=name)

    assert not any(boundary_sublayer_flags)
    if stage_is_frozen and not model.post_process:
        # The whole stage lies before the boundary, so the schedule skips its backward.
        # The last stage still feeds its trainable final norm.
        assert boundary_output_flags and not any(boundary_output_flags)
    # Under recompute, hooks also fire inside checkpoints and during replay; check eager only.
    first_stage = parallel_state.is_pipeline_first_stage()
    if not recompute and not first_stage and layers[0].layer_number - 1 < first_trainable:
        assert reference_sublayer_flags and all(reference_sublayer_flags)
