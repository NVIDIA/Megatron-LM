# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared input/output weights at language-model module boundaries."""

import copy

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    Placements,
    SchedulePolicy,
    fully_shard,
    fully_shard_context,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(params=["gpt", "hybrid"])
def tied_model(request):
    Utils.initialize_model_parallel(1, 1)
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend.unfused,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    kwargs = dict(
        config=config,
        vocab_size=128,
        max_sequence_length=4,
        share_embeddings_and_output_weights=True,
        position_embedding_type="rope",
    )
    if request.param == "gpt":
        model = GPTModel(
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(), **kwargs
        )
    else:
        model = HybridModel(hybrid_stack_spec=hybrid_stack_spec, hybrid_layer_pattern="*", **kwargs)
    yield model.cuda()
    Utils.destroy_model_parallel()


def test_tied_output_preserves_checkpoint_keys(tied_model):
    """Both consumers share one parameter; checkpoints keep the embedding-only name."""
    model = tied_model
    assert model.output_layer.weight is model.embedding.word_embeddings.weight
    state = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in model.state_dict().items()
    }
    assert "output_layer.weight" not in state
    assert "output_layer.weight" not in model.sharded_state_dict(
        metadata={"dp_cp_group": model.pg_collection.dp_cp}
    )
    with torch.no_grad():
        model.embedding.word_embeddings.weight.zero_()
    model.load_state_dict(state, strict=True)
    torch.testing.assert_close(model.output_layer.weight, state["embedding.word_embeddings.weight"])
    assert model.output_layer.weight is model.embedding.word_embeddings.weight


@pytest.mark.mfsdp
@pytest.mark.parametrize("training", [False, True])
def test_tied_output_unshards_before_weight_lookup(tied_model, training):
    """A head-only demand hook works when the scheduler bypasses the root forward hook."""
    model = tied_model
    model.train(training)
    reference = copy.deepcopy(model)
    device = torch.device("cuda", torch.cuda.current_device())
    mesh = init_device_mesh("cuda", (torch.distributed.get_world_size(),))
    placements = Placements([0], [Shard(0)], [Shard(0)], [Shard(0)])
    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(model.decoder, mesh=mesh, placements=placements)
        fully_shard(
            model,
            mesh=mesh,
            placements=placements,
            schedule_policy=SchedulePolicy(register_hooks=False),
        )

    def unshard_head(_module, _args, kwargs):
        assert kwargs.get("weight") is None
        model.unshard()
        assert model.output_layer.weight is model.embedding.word_embeddings.weight

    model.output_layer.register_forward_pre_hook(unshard_head, with_kwargs=True)
    # Supplying decoder_input skips the embedding lookup, leaving root weights sharded
    # until the output head's pre-hook. A previously captured weight would then be stale.
    positions = torch.arange(4, device=device).unsqueeze(0).expand(2, -1)
    for _ in range(2):
        hidden = torch.randn(4, 2, 64, device=device)
        with torch.no_grad():
            expected = reference(None, positions, None, decoder_input=hidden)
            actual = model(None, positions, None, decoder_input=hidden)
        torch.testing.assert_close(actual, expected)
        model.reshard()
        assert model.output_layer.weight is model.embedding.word_embeddings.weight

    hidden = torch.randn(4, 2, 64, device=device, requires_grad=True)
    logits, _ = model.output_layer(hidden, detach_weight=True)
    logits.sum().backward()
    assert hidden.grad is not None
    assert model.output_layer.weight.grad is None
    model.reshard()
