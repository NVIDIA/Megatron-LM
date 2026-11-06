# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Language model loss when the output layer gathers the logits over tensor parallelism.

`parallel_output=False` and `runtime_gather_output=True` make the output layer all-gather the
logits over the vocabulary. The vocab-parallel cross entropy reads the last dimension as this
rank's vocabulary shard, so a model must hand it only its own shard of the gathered logits. Given
the gathered logits, the loss came out log(tp_size) too high and the gradients were wrong
(#7452, finding 2).
"""

import pytest
import torch

from megatron.core.models.bert.bert_layer_specs import bert_layer_local_spec
from megatron.core.models.bert.bert_model import BertModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.T5.t5_model import T5Model
from megatron.core.models.T5.t5_spec import (
    get_t5_decoder_with_local_block_spec,
    get_t5_encoder_with_local_block_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

TP_SIZE = 2
VOCAB_SIZE = 64
SEQ_LENGTH = 8
MICRO_BATCH_SIZE = 2


def _build_model(model_type, parallel_output):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=TP_SIZE,
        attention_backend=AttnBackend.local,
    )
    kwargs = dict(
        config=config,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LENGTH,
        parallel_output=parallel_output,
    )
    if model_type == "gpt":
        return GPTModel(transformer_layer_spec=get_gpt_layer_local_spec(), **kwargs)
    if model_type == "hybrid":
        hybrid_stack_spec = ModuleSpec(
            module=HybridStack,
            submodules=HybridStackSubmodules(attention_layer=get_gpt_layer_local_spec()),
        )
        return HybridModel(hybrid_stack_spec=hybrid_stack_spec, hybrid_layer_pattern="*", **kwargs)
    if model_type == "bert":
        return BertModel(
            transformer_layer_spec=bert_layer_local_spec,
            num_tokentypes=0,
            add_binary_head=False,
            **kwargs,
        )
    return T5Model(
        encoder_config=config,
        transformer_encoder_layer_spec=get_t5_encoder_with_local_block_spec(1),
        transformer_decoder_layer_spec=get_t5_decoder_with_local_block_spec(1),
        **kwargs,
    )


def _loss_and_grads(model, model_type, runtime_gather_output=None):
    generator = torch.Generator().manual_seed(1234)
    shape = (MICRO_BATCH_SIZE, SEQ_LENGTH)
    input_ids = torch.randint(0, VOCAB_SIZE, shape, generator=generator).cuda()
    labels = torch.randint(0, VOCAB_SIZE, shape, generator=generator).cuda()
    causal_mask = torch.ones((SEQ_LENGTH, SEQ_LENGTH), dtype=torch.bool).triu(diagonal=1)
    causal_mask = causal_mask.expand(MICRO_BATCH_SIZE, 1, SEQ_LENGTH, SEQ_LENGTH).cuda()

    model.zero_grad(set_to_none=True)
    if model_type in ("gpt", "hybrid"):
        loss = model(
            input_ids=input_ids,
            position_ids=torch.arange(SEQ_LENGTH).repeat(MICRO_BATCH_SIZE, 1).cuda(),
            attention_mask=causal_mask,
            labels=labels,
            runtime_gather_output=runtime_gather_output,
        )
    elif model_type == "bert":
        loss, _ = model(
            input_ids=input_ids,
            attention_mask=torch.ones(shape, dtype=torch.bool).cuda(),
            lm_labels=labels,
        )
    else:
        no_mask = torch.zeros_like(causal_mask)
        loss = model(
            encoder_input_ids=input_ids,
            decoder_input_ids=input_ids,
            encoder_attn_mask=no_mask,
            decoder_attn_mask=causal_mask,
            encoder_decoder_attn_mask=no_mask,
            lm_labels=labels,
        )
    loss.sum().backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    return loss.detach(), grads


def _assert_same_loss_and_grads(actual, expected):
    (loss, grads), (expected_loss, expected_grads) = actual, expected
    torch.testing.assert_close(loss, expected_loss)
    assert grads.keys() == expected_grads.keys()
    for name, grad in grads.items():
        torch.testing.assert_close(grad, expected_grads[name], msg=lambda msg: f"{name}: {msg}")


class TestGatheredLogitsLoss:
    """The loss and gradients with gathered logits match those with sharded logits."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=TP_SIZE)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("model_type", ["gpt", "hybrid", "bert", "t5"])
    def test_parallel_output_false(self, model_type):
        sharded_model = _build_model(model_type, parallel_output=True).cuda()
        gathered_model = _build_model(model_type, parallel_output=False).cuda()
        gathered_model.load_state_dict(sharded_model.state_dict())
        assert gathered_model.output_layer.gather_output

        _assert_same_loss_and_grads(
            _loss_and_grads(gathered_model, model_type), _loss_and_grads(sharded_model, model_type)
        )

    @pytest.mark.parametrize("model_type", ["gpt", "hybrid"])
    def test_runtime_gather_output(self, model_type):
        model = _build_model(model_type, parallel_output=True).cuda()

        _assert_same_loss_and_grads(
            _loss_and_grads(model, model_type, runtime_gather_output=True),
            _loss_and_grads(model, model_type),
        )
