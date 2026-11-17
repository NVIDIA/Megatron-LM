# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native MIMO packed metadata through real SSA forward and backward."""

import copy

import pytest
import torch

from megatron.core.models.mimo.partition.utils import PartitionAdapter, PartitionConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import (
    simplified_sparse_attention as ssa,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import flatten_batch_for_packed_sequences
from tests.unit_tests.test_utilities import Utils


def _language_batch(raw):
    """A MIMO language batch from a packed one: flattened metadata, tokens as input IDs."""
    batch = flatten_batch_for_packed_sequences(dict(raw))
    batch["input_ids"] = batch.pop("tokens")
    return batch


@pytest.mark.parametrize("phase1", [False, True])
@pytest.mark.parametrize("source_padding", [False, True])
def test_mimo_idm_image_and_padding_cp2_matches_cp1(monkeypatch, phase1, source_padding):
    """Compare both-phase outputs, auxiliary loss, input and indexer-weight gradients.

    Image embeddings are already assembled, as covered by MIMO's embedding-alignment
    tests. This gate exercises the real attention kernels and collectives after the
    MIMO partition adapter, not a mocked SSA operation.
    """
    from megatron.core.transformer.experimental_attention_variant.simplified_sparse_attention_cudnn import (  # pylint: disable=line-too-long
        bind_cudnn_sparse_attention,
    )

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("The cuDNN kernels require SM100 or SM103")
    if Utils.world_size < 2:
        pytest.skip("CP2 requires at least two processes")
    try:
        bind_cudnn_sparse_attention(dense_teacher=phase1)
    except ImportError as error:
        pytest.skip(str(error))

    Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(453)
        groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        cp_rank = groups.cp.rank()
        torch.manual_seed(57)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            ffn_hidden_size=512,
            num_attention_heads=16,
            num_query_groups=1,
            kv_channels=256,
            params_dtype=torch.bfloat16,
            bf16=True,
            use_cpu_initialization=True,
            experimental_attention_variant="ssa",
            dsa_indexer_topk=512,
            dsa_indexer_loss_coeff=0.03,
            ssa_train_indexer_only=phase1,
            attention_dropout=0.0,
            context_parallel_size=2,
            cp_comm_type="allgather",
        )
        reference_config = copy.copy(config)
        reference_config.context_parallel_size = 1
        reference_groups = ProcessGroupCollection()
        reference_groups.tp = groups.tp
        reference_groups.cp = None
        local_core = ssa.SimplifiedSparseAttentionCore(
            config, 1, AttnMaskType.causal, "self", groups, cp_comm_type="allgather"
        ).cuda()
        reference_core = ssa.SimplifiedSparseAttentionCore(
            reference_config, 1, AttnMaskType.causal, "self", reference_groups
        ).cuda()
        reference_core.load_state_dict(local_core.state_dict())
        logged_losses = []
        monkeypatch.setattr(
            ssa.DSAIndexerLossLoggingHelper,
            "save_loss_to_tracker",
            lambda **kwargs: logged_losses.append(kwargs["loss"].detach()),
        )
        monkeypatch.setattr(ssa.DSAIndexerLossAutoScaler, "main_loss_backward_scale", None)

        # Native producers retain logical document lengths separately from storage
        # offsets. An image row excluded from LM loss is still a valid query.
        tokens = torch.arange(128, device="cuda").reshape(1, -1)
        tokens[0, 1] = 999  # One real image row with no LM prediction objective.
        padding = torch.zeros(1, 128, dtype=torch.bool, device="cuda")
        if source_padding:
            padding[0, 61:64] = True
            padding[0, 123:] = True
        lm_mask = (~padding).float()
        lm_mask[0, 1] = 0
        raw = {
            "tokens": tokens,
            "labels": tokens,
            "loss_mask": lm_mask,
            "position_ids": torch.arange(64, device="cuda").repeat(2).reshape(1, -1),
            "cu_seqlens": torch.tensor([[0, 64, 128]], dtype=torch.int32, device="cuda"),
            "max_seqlen": torch.tensor([64], dtype=torch.int32, device="cuda"),
        }
        if source_padding:
            raw["cu_seqlens_padded"] = raw["cu_seqlens"]
            raw["cu_seqlens"] = torch.tensor([[0, 61, 120]], dtype=torch.int32, device="cuda")
        native = _language_batch(raw)
        hidden = torch.randn(128, 1, 256, device="cuda", dtype=torch.bfloat16)
        # Stand in for the independently covered image-embedding replacement.
        hidden[1] = torch.randn_like(hidden[1])

        def partition(cp_group, use_cp):
            return PartitionAdapter(
                PartitionConfig(
                    seq_parallel=False,
                    use_cp=use_cp,
                    tp_comm_overlap=False,
                    max_seq_len=128,
                    cp_group=cp_group,
                    pad_packed_documents=True,
                )
            ).partition(
                embeddings=hidden,
                input_ids=native["input_ids"],
                position_ids=native["position_ids"],
                labels=native["labels"],
                loss_mask=native["loss_mask"],
                mtp_input_mask=None,
                cu_seqlens=native["cu_seqlens"],
                cu_seqlens_padded=native.get("cu_seqlens_padded"),
                max_seqlen=native["max_seqlen"],
            )

        local_batch = partition(groups.cp, True)
        global_batch = partition(None, False)
        positions = torch.cat(
            [
                torch.arange(start + chunk * 16, start + (chunk + 1) * 16, device="cuda")
                for start in (0, 64)
                for chunk in (cp_rank, 3 - cp_rank)
            ]
        )
        if cp_rank == 0:
            assert local_batch.get_batch()["loss_mask"][0, 1] == 0
        assert "padding_mask" not in native
        assert local_batch.get_batch().get("padding_mask") is None
        assert local_batch.get_packed_seq_params().cu_seqlens_q.tolist() == (
            [0, 61, 120] if source_padding else [0, 64, 128]
        )
        assert local_batch.get_packed_seq_params().cu_seqlens_q_padded.tolist() == [0, 64, 128]
        global_inputs = [
            torch.randn(128, 1, heads, 256, device="cuda", dtype=torch.bfloat16)
            for heads in (16, 1, 1)
        ]
        local_inputs = [tensor[positions].detach().requires_grad_() for tensor in global_inputs]
        refs = [tensor.detach().requires_grad_() for tensor in global_inputs]
        output = local_core(
            *(tensor[:, 0] for tensor in local_inputs),
            attention_mask=None,
            packed_seq_params=local_batch.get_packed_seq_params(),
            indexer_input=local_batch.get_batch()["decoder_input"],
        ).reshape(64, 1, -1)
        expected = reference_core(
            *(tensor[:, 0] for tensor in refs),
            attention_mask=None,
            packed_seq_params=global_batch.get_packed_seq_params(),
            indexer_input=global_batch.get_batch()["decoder_input"],
        ).reshape(128, 1, -1)
        torch.testing.assert_close(
            output.float(), expected[positions].float(), rtol=2e-2, atol=2e-2
        )
        total_loss = logged_losses[0].clone()
        torch.distributed.all_reduce(total_loss, group=groups.cp)
        torch.testing.assert_close(total_loss, logged_losses[1], rtol=2e-2, atol=2e-4)

        upstream = torch.randn_like(expected)
        (output * upstream[positions]).sum().backward()
        (expected * upstream).sum().backward()
        for actual, reference in zip(local_inputs, refs):
            if phase1:
                assert actual.grad is None and reference.grad is None
            else:
                torch.testing.assert_close(
                    actual.grad.float(), reference.grad[positions].float(), rtol=4e-2, atol=4e-2
                )
        for actual, reference in zip(
            local_core.indexer.parameters(), reference_core.indexer.parameters()
        ):
            assert actual.grad is not None and reference.grad is not None
            gradient = actual.grad.float()
            torch.distributed.all_reduce(gradient, group=groups.cp)
            torch.testing.assert_close(gradient, reference.grad.float(), rtol=4e-2, atol=4e-3)
    finally:
        Utils.destroy_model_parallel()
