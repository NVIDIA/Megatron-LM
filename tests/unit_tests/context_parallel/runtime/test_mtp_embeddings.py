# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real packed rolling and embedding gradients with a CP1-built MTP module."""

import os
import types

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_dynamic_data_context_parallel_groups,
    get_tensor_model_parallel_group,
)
from megatron.core.transformer import multi_token_prediction as mtp_module
from megatron.core.transformer.multi_token_prediction import (
    MTPLossLoggingHelper,
    MultiTokenPredictionBlock,
    process_mtp_loss,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import destroy_global_vars
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200


@pytest.fixture(autouse=True)
def cleanup_runtime_mtp():
    os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    yield
    Utils.destroy_model_parallel()
    destroy_global_vars()
    destroy_num_microbatches_calculator()


@pytest.mark.parametrize("with_mask", [False, True])
def test_block_rolls_precomputed_embeddings_with_runtime_cp(with_mask):
    """A CP1-built block must roll embeddings and validity on each microbatch's group."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2 or world_size % 2:
        pytest.skip("Runtime CP2 rolling requires an even number of distributed ranks")

    Utils.initialize_model_parallel(dynamic_context_parallel=True)
    build_group = get_context_parallel_group()
    assert build_group.size() == 1
    config = TransformerConfig(num_layers=2, hidden_size=1, num_attention_heads=1, mtp_num_layers=2)
    seen = []

    class _CaptureLayer:
        def __call__(
            self, hidden_states, input_ids, position_ids, padding_mask, decoder_input, **kwargs
        ):
            seen.append((decoder_input.clone(), padding_mask))
            return hidden_states, input_ids, position_ids, padding_mask, None

    block = types.SimpleNamespace(
        config=config,
        training=True,
        vp_stage=None,
        pp_rank=0,
        mtp_use_repeated_layer=False,
        cp_group=build_group,
        tp_group=get_tensor_model_parallel_group(),
        sequence_parallel=False,
        layers=[_CaptureLayer(), _CaptureLayer()],
    )
    lengths = (8, 16)
    full_embeddings = torch.arange(1, 25, dtype=torch.float32, device="cuda").view(24, 1, 1)
    full_validity = full_embeddings[:, :, 0].remainder(3) != 0
    cu_seqlens = torch.tensor([0, 8, 24], dtype=torch.int32, device="cuda")

    def shard(tensor, group):
        pieces = []
        for sequence in tensor.split(lengths):
            chunks = sequence.chunk(2 * group.size())
            pieces.extend((chunks[group.rank()], chunks[2 * group.size() - group.rank() - 1]))
        return torch.cat(pieces).contiguous()

    def shifted(tensor, depth):
        # Full-sequence oracle: advance within each document and zero its trailing slots.
        return torch.cat(
            [
                torch.cat((seq[depth:], torch.zeros_like(seq[:depth])))
                for seq in tensor.split(lengths)
            ]
        )

    for cp_size in (1, 2, 1):
        group = get_dynamic_data_context_parallel_groups(group_size=cp_size)
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max(lengths),
            max_seqlen_kv=max(lengths),
            local_cp_size=cp_size,
            cp_group=group,
        )
        embeddings = shard(full_embeddings, group)
        padding_mask = ~shard(full_validity, group).transpose(0, 1) if with_mask else None
        seen.clear()
        MultiTokenPredictionBlock.forward(
            block,
            input_ids=None,
            position_ids=None,
            hidden_states=torch.ones_like(embeddings),
            decoder_input=embeddings,
            attention_mask=None,
            padding_mask=padding_mask,
            packed_seq_params=packed,
        )
        assert len(seen) == config.mtp_num_layers
        for depth, (actual_embeddings, actual_mask) in enumerate(seen, start=1):
            torch.testing.assert_close(
                actual_embeddings, shard(shifted(full_embeddings, depth), group)
            )
            if with_mask:
                expected_mask = ~shard(shifted(full_validity, depth), group).transpose(0, 1)
                torch.testing.assert_close(actual_mask, expected_mask)
            else:
                assert actual_mask is None
        assert block.cp_group is build_group


def test_runtime_cp_packed_preprocess_forward_backward_parity():
    """Runtime CP metadata matches an MTP layer bound directly to the same real group."""
    runtime_cp_size = 2
    if int(os.environ.get("WORLD_SIZE", "1")) < runtime_cp_size:
        pytest.skip("Runtime MTP CP parity requires at least two distributed ranks")

    torch.manual_seed(123)
    Utils.initialize_model_parallel(dynamic_context_parallel=True)
    config = TransformerConfig(
        mtp_num_layers=2,
        num_layers=4,
        hidden_size=64,
        num_attention_heads=8,
        use_cpu_initialization=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        mtp_use_repeated_layer=False,
        context_parallel_size=1,
    )
    mtp_block_spec = get_gpt_mtp_block_spec(
        config=config, spec=get_gpt_layer_local_spec(), use_transformer_engine=False
    )
    runtime_group = get_dynamic_data_context_parallel_groups(group_size=runtime_cp_size)

    reference_block = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec).cuda()
    dynamic_block = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec).cuda()
    dynamic_block.load_state_dict(reference_block.state_dict())
    reference_layer = reference_block.layers[0]
    dynamic_layer = dynamic_block.layers[0]
    reference_layer.cp_group = runtime_group

    cp_rank = torch.distributed.get_rank(group=runtime_group)
    if cp_rank == 0:
        input_ids = torch.tensor(
            [[1, 2, 7, 8, 11, 12, 13, 20, 21, 22]], dtype=torch.int64, device="cuda"
        )
    else:
        input_ids = torch.tensor(
            [[3, 4, 5, 6, 14, 15, 16, 17, 18, 19]], dtype=torch.int64, device="cuda"
        )
    position_ids = torch.arange(input_ids.size(1), device="cuda").unsqueeze(0)
    mtp_input_mask = (input_ids != 3) & (input_ids != 14)
    cu_seqlens = torch.tensor([0, 8, 20], dtype=torch.int32, device="cuda")

    reference_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=6,
        max_seqlen_kv=6,
        qkv_format="thd",
    )
    dynamic_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=6,
        max_seqlen_kv=6,
        qkv_format="thd",
        local_cp_size=runtime_cp_size,
        cp_group=runtime_group,
    )

    torch.manual_seed(99 + cp_rank)
    hidden = torch.randn(
        input_ids.size(1), 1, config.hidden_size, device="cuda", dtype=torch.float32
    )

    def embedding(input_ids, position_ids):
        del position_ids
        return (
            input_ids.transpose(0, 1)
            .unsqueeze(-1)
            .expand(-1, -1, config.hidden_size)
            .to(dtype=torch.float32)
            / 100.0
        )

    def run(layer, packed_seq_params, hidden_states):
        rolled_ids, _, _, rolled_mask, decoder_input, rolled_hidden = layer._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            embedding=embedding,
            hidden_states=hidden_states,
            packed_seq_params=packed_seq_params,
            mtp_input_mask=mtp_input_mask,
        )
        output = layer._concat_embeddings(rolled_hidden, decoder_input)
        output.square().sum().backward()
        return rolled_ids, rolled_mask, output.detach(), hidden_states.grad.detach()

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    dynamic_hidden = hidden.detach().clone().requires_grad_(True)
    reference_result = run(reference_layer, reference_params, reference_hidden)
    dynamic_result = run(dynamic_layer, dynamic_params, dynamic_hidden)

    for reference_value, dynamic_value in zip(reference_result, dynamic_result, strict=True):
        torch.testing.assert_close(reference_value, dynamic_value)
    for (reference_name, reference_param), (dynamic_name, dynamic_param) in zip(
        reference_layer.named_parameters(), dynamic_layer.named_parameters(), strict=True
    ):
        assert reference_name == dynamic_name
        assert (reference_param.grad is None) == (
            dynamic_param.grad is None
        ), f"MTP runtime-CP gradient participation differs for {reference_name}"
        if reference_param.grad is None:
            continue
        torch.testing.assert_close(
            reference_param.grad,
            dynamic_param.grad,
            msg=lambda msg, param_name=reference_name: (
                f"Mismatch in MTP runtime-CP gradient {param_name}: {msg}"
            ),
        )


def test_process_mtp_per_token_loss_uses_explicit_metric_group(monkeypatch):
    """MTP metric reduction must use the language model's supplied data group."""
    per_token = True
    metric_avg_group = object()
    captured = {}

    def unexpected_global_read(*args, **kwargs):
        raise AssertionError("global data-parallel state must not be read")

    def capture_metrics(*args, **kwargs):
        captured["avg_group"] = kwargs["avg_group"]
        captured["loss"] = args[0]
        captured["num_tokens"] = kwargs["num_tokens"]

    monkeypatch.setattr(
        mtp_module.parallel_state, "get_data_parallel_group", unexpected_global_read
    )
    monkeypatch.setattr(MTPLossLoggingHelper, "save_metrics_to_tracker", capture_metrics)

    config = TransformerConfig(
        mtp_num_layers=1,
        mtp_loss_scaling_factor=1.0,
        num_layers=2,
        hidden_size=1,
        num_attention_heads=1,
        use_cpu_initialization=True,
        calculate_per_token_loss=per_token,
    )
    seq_len = 4
    hidden_states = torch.ones(2 * seq_len, 1, 1)

    process_mtp_loss(
        hidden_states=hidden_states,
        labels=torch.zeros(1, seq_len, dtype=torch.long),
        loss_mask=torch.ones(1, seq_len),
        output_layer=lambda hidden, **kwargs: (hidden, None),
        output_weight=None,
        is_training=True,
        compute_language_model_loss=lambda labels, logits: torch.ones_like(
            labels, dtype=logits.dtype
        ),
        config=config,
        metric_avg_group=metric_avg_group,
    )

    assert captured["avg_group"] is metric_avg_group
    assert captured["loss"].item() == (seq_len - 1 if per_token else 1)
    if per_token:
        assert captured["num_tokens"].item() == seq_len - 1
    else:
        assert captured["num_tokens"] is None
