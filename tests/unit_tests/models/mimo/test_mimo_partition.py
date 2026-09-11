# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

'''
WORLD_SIZE=1 LOCAL_RANK=0 python -m torch.distributed.run \
    --nproc_per_node=1 -m pytest \
    tests/unit_tests/models/mimo/test_mimo_partition.py -v
'''

import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.context_parallel import ContextParallelBatch
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.mimo.partition.utils import PartitionAdapter, PartitionConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.experimental
class TestPartitionConfig:
    """Tests for PartitionConfig dataclass and factory method."""

    def test_from_mp_config_invalid_type_raises(self):
        with pytest.raises(TypeError, match="mp must be a ModelParallelConfig instance"):
            PartitionConfig.from_mp_config("not_a_config", max_seq_len=128)

    def test_from_mp_config_no_parallelism(self):
        mp = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            context_parallel_size=1,
            sequence_parallel=False,
        )
        with patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=1):
            cfg = PartitionConfig.from_mp_config(mp, max_seq_len=512)
        assert cfg.use_cp is False
        assert cfg.seq_parallel is False
        assert cfg.cp_group is None
        assert cfg.tp_group is None
        assert cfg.max_seq_len == 512

    def test_from_mp_config_kv_format_thd(self):
        mp = TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=4)
        with patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=1):
            cfg = PartitionConfig.from_mp_config(mp, max_seq_len=512, kv_format='thd')
        assert cfg.kv_format == 'thd'

    def test_from_mp_config_explicit_cp_group(self):
        mock_cp_group = MagicMock()
        mp = TransformerConfig(
            num_layers=1, hidden_size=64, num_attention_heads=4, context_parallel_size=2
        )
        with patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=2):
            cfg = PartitionConfig.from_mp_config(mp, max_seq_len=512, cp_group=mock_cp_group)
        assert cfg.use_cp is True
        assert cfg.cp_group is mock_cp_group

    def test_from_mp_config_explicit_tp_group(self):
        mock_tp_group = MagicMock()
        mock_tp_cp_group = MagicMock()
        mp = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )
        with patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=1):
            cfg = PartitionConfig.from_mp_config(
                mp, max_seq_len=512, tp_group=mock_tp_group, tp_cp_group=mock_tp_cp_group
            )
        assert cfg.seq_parallel is True
        assert cfg.tp_group is mock_tp_group
        assert cfg.tp_cp_group is mock_tp_cp_group

    def test_from_mp_config_auto_fetch_cp_group(self):
        mock_group = MagicMock()
        mp = TransformerConfig(
            num_layers=1, hidden_size=64, num_attention_heads=4, context_parallel_size=2
        )
        with (
            patch(
                'megatron.core.models.mimo.partition.utils.get_context_parallel_group',
                return_value=mock_group,
            ),
            patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=2),
        ):
            cfg = PartitionConfig.from_mp_config(mp, max_seq_len=512)
        assert cfg.cp_group is mock_group

    def test_from_mp_config_auto_fetch_tp_group(self):
        mock_group = MagicMock()
        mp = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )
        with (
            patch(
                'megatron.core.models.mimo.partition.utils.get_tensor_model_parallel_group',
                return_value=mock_group,
            ),
            patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=1),
        ):
            cfg = PartitionConfig.from_mp_config(mp, max_seq_len=512)
        assert cfg.tp_group is mock_group


@pytest.mark.experimental
class TestPartitionAdapter:
    """Tests for PartitionAdapter.partition()."""

    def _make_cfg(
        self,
        use_cp=False,
        seq_parallel=False,
        tp_comm_overlap=False,
        max_seq_len=128,
        cp_group=None,
        tp_group=None,
    ):
        return PartitionConfig(
            use_cp=use_cp,
            seq_parallel=seq_parallel,
            tp_comm_overlap=tp_comm_overlap,
            max_seq_len=max_seq_len,
            cp_group=cp_group,
            tp_group=tp_group,
        )

    def _make_tensors(self, B=2, S=8, H=16):
        # Embeddings are sequence-first (S, B, H); labels/loss_mask are (B, S).
        embeddings = torch.rand(S, B, H)
        labels = torch.randint(0, 100, (B, S))
        loss_mask = torch.ones(B, S)
        return embeddings, labels, loss_mask

    def test_noop_when_both_disabled(self):
        """With neither CP nor SP active, partition() is a pure passthrough.

        Embeddings are already sequence-first (S, B, H), so with no collectives the inputs are
        returned untouched while packed metadata still follows the shared preparation path.
        """
        cfg = self._make_cfg(use_cp=False, seq_parallel=False)
        adapter = PartitionAdapter(cfg)
        embeddings, labels, loss_mask = self._make_tensors(B=2, S=8, H=16)
        cp_batch = adapter.partition(embeddings, None, None, labels, loss_mask, None)
        batch = cp_batch.get_batch()
        assert batch["decoder_input"] is embeddings
        assert batch["labels"] is labels
        assert batch["loss_mask"] is loss_mask
        assert cp_batch.get_packed_seq_params() is None

    def test_seq_not_divisible_raises(self):
        mock_cp_group = MagicMock()
        cfg = self._make_cfg(use_cp=True, max_seq_len=7, cp_group=mock_cp_group)
        adapter = PartitionAdapter(cfg)
        embeddings = torch.rand(7, 2, 16)  # seq-first [S, B, H]; 7 % (2*2) != 0
        labels = torch.randint(0, 100, (2, 7))
        loss_mask = torch.ones(2, 7)
        with (
            patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=2),
            pytest.raises(AssertionError, match="divisible"),
        ):
            adapter.partition(embeddings, None, None, labels, loss_mask, None)

    def test_tp_comm_overlap_seq_len_assertion(self):
        mock_tp_group = MagicMock()
        cfg = self._make_cfg(
            seq_parallel=True, tp_comm_overlap=True, max_seq_len=16, tp_group=mock_tp_group
        )
        adapter = PartitionAdapter(cfg)
        # S=8 (seq-first [S, B, H]) but max_seq_len=16 → assertion fires
        embeddings = torch.rand(8, 2, 16)
        labels = torch.randint(0, 100, (2, 8))
        loss_mask = torch.ones(2, 8)
        with (
            patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=2),
            pytest.raises(AssertionError, match="TP Comm overlap"),
        ):
            adapter.partition(embeddings, None, None, labels, loss_mask, None)

    def test_thd_format_skips_divisibility_check(self):
        """Packed-sequence metadata bypasses the dense divisibility assertion."""
        mock_cp_group = MagicMock()
        cfg = self._make_cfg(use_cp=True, max_seq_len=7, cp_group=mock_cp_group)
        adapter = PartitionAdapter(cfg)
        embeddings = torch.rand(7, 2, 16)  # seq-first; len=7 not divisible by cp*2, THD skips check
        labels = torch.randint(0, 100, (2, 7))
        loss_mask = torch.ones(2, 7)
        cu_seqlens = torch.tensor([[0, 4, 7]], dtype=torch.int32)
        cp_batch = ContextParallelBatch.from_single_layout(
            "zigzag",
            {
                "decoder_input": embeddings.transpose(0, 1)[:, :4],
                "labels": labels[:, :4],
                "loss_mask": loss_mask[:, :4],
            },
            MagicMock(spec=PackedSeqParams),
        )
        with (
            patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=2),
            patch(
                'megatron.core.models.mimo.partition.utils.get_batches_on_this_cp_rank',
                return_value=cp_batch,
            ),
        ):
            # Should NOT raise AssertionError about divisibility
            result = adapter.partition(
                embeddings,
                None,
                None,
                labels,
                loss_mask,
                None,
                cu_seqlens=cu_seqlens,
                max_seqlen=torch.tensor([4], dtype=torch.int32),
            )
        assert result.get_batch()["decoder_input"] is not None

    def test_none_embeddings_skips_shard_factor_check(self):
        """When embeddings is None, the divisibility check is skipped (non-first PP stage)."""
        mock_cp_group = MagicMock()
        cfg = self._make_cfg(use_cp=True, max_seq_len=7, cp_group=mock_cp_group)
        adapter = PartitionAdapter(cfg)
        labels = torch.randint(0, 100, (2, 7))
        loss_mask = torch.ones(2, 7)
        cp_batch = ContextParallelBatch.from_single_layout(
            "zigzag", {'labels': labels[:, :4], 'loss_mask': loss_mask[:, :4]}, None
        )
        with (
            patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=2),
            patch(
                'megatron.core.models.mimo.partition.utils.get_batches_on_this_cp_rank',
                return_value=cp_batch,
            ),
        ):
            result = adapter.partition(None, None, None, labels, loss_mask, None)
        batch = result.get_batch()
        assert batch.get("decoder_input") is None
        assert batch["labels"].shape == (2, 4)
        assert batch["loss_mask"].shape == (2, 4)


@pytest.mark.experimental
class TestPartitionAdapterContextParallelBatch:
    """Tests for constructing MIMO's dual-layout CP batch."""

    def test_partitions_all_language_inputs_together(self):
        cp_group = MagicMock()
        tp_group = MagicMock()
        tp_cp_group = MagicMock()
        cfg = PartitionConfig(
            use_cp=True,
            seq_parallel=True,
            tp_comm_overlap=False,
            max_seq_len=8,
            linear_cp_layout="contiguous",
            attention_cp_layout="zigzag",
            cp_group=cp_group,
            tp_group=tp_group,
            tp_cp_group=tp_cp_group,
        )
        adapter = PartitionAdapter(cfg)
        embeddings, labels, loss_mask = TestPartitionAdapter()._make_tensors(S=8)
        input_ids = torch.arange(8).view(1, 8).expand(2, -1)
        position_ids = torch.stack((input_ids, input_ids + 10, input_ids + 20))
        mtp_input_mask = input_ids != 3
        cu_seqlens = torch.tensor([[0, 8, 16]], dtype=torch.int32)
        max_seqlen = torch.tensor([8], dtype=torch.int32)

        boundary_batch = {
            "tokens": input_ids[:, :4],
            "position_ids": position_ids.movedim(0, -1)[:, :4],
            "labels": labels[:, :4],
            "loss_mask": loss_mask[:, :4],
            "mtp_input_mask": mtp_input_mask[:, :4],
            "decoder_input": embeddings.transpose(0, 1)[:, :4],
        }
        zigzag_batch = {
            "tokens": input_ids[:, 4:],
            "position_ids": position_ids.movedim(0, -1)[:, 4:],
            "labels": labels[:, 4:],
            "loss_mask": loss_mask[:, 4:],
            "mtp_input_mask": mtp_input_mask[:, 4:],
            "decoder_input": embeddings.transpose(0, 1)[:, 4:],
        }
        local_packed = MagicMock(spec=PackedSeqParams)
        cp_batch = ContextParallelBatch(
            boundary_layout="contiguous",
            batches_by_layout={"contiguous": boundary_batch, "zigzag": zigzag_batch},
            packed_seq_params_by_layout={
                "contiguous": local_packed,
                "zigzag": MagicMock(spec=PackedSeqParams),
            },
            thd_plan=MagicMock(),
        )
        local_embeddings = embeddings[:2]

        with (
            patch(
                "megatron.core.models.mimo.partition.utils.get_batches_on_this_cp_rank",
                return_value=cp_batch,
            ) as get_cp_batches,
            patch(
                "megatron.core.models.mimo.partition.utils."
                "tensor_parallel.scatter_to_sequence_parallel_region",
                return_value=local_embeddings,
            ) as scatter,
        ):
            result = adapter.partition(
                embeddings=embeddings,
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                loss_mask=loss_mask,
                mtp_input_mask=mtp_input_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        (unsharded_batch,) = get_cp_batches.call_args.args
        call_kwargs = get_cp_batches.call_args.kwargs
        assert call_kwargs["boundary_layout"] == "contiguous"
        assert call_kwargs["additional_layouts"] == {"zigzag"}
        assert call_kwargs["tp_cp_group"] is tp_cp_group
        assert unsharded_batch["decoder_input"].shape == (2, 8, 16)
        assert unsharded_batch["position_ids"].shape == (2, 8, 3)
        assert unsharded_batch["mtp_input_mask"] is mtp_input_mask
        assert unsharded_batch["cu_seqlens"] is cu_seqlens
        assert unsharded_batch["max_seqlen"] is max_seqlen
        assert result is cp_batch
        assert boundary_batch["decoder_input"] is local_embeddings
        assert boundary_batch["position_ids"].shape == (3, 2, 4)
        assert result.get_packed_seq_params() is local_packed
        assert "decoder_input" not in zigzag_batch
        scatter.assert_called_once()


def _expected_cp_zigzag_shard(tensor: torch.Tensor, cp_size: int, cp_rank: int) -> torch.Tensor:
    """Reconstruct the CP zigzag shard of ``tensor`` along the sequence dim (dim 1).

    Mirrors ``_get_batch_on_this_cp_rank_per_sequence_balancing``: the sequence is split into
    ``2 * cp_size`` equal chunks and rank ``r`` keeps chunks ``r`` and
    ``2*cp_size - r - 1`` (concatenated in that order). Implemented independently
    here so the real-distributed assertions do not lean on the production helper.
    """
    if cp_size == 1:
        return tensor
    chunks = list(torch.chunk(tensor, 2 * cp_size, dim=1))
    return torch.cat([chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]], dim=1)


@pytest.mark.experimental
@pytest.mark.skipif(
    int(os.environ.get('WORLD_SIZE', '1')) != 8,
    reason="Real MIMO CP/SP sharding tests require an 8-GPU world",
)
class TestPartitionAdapterRealDistributed:
    """Real 8-GPU tests for ``PartitionAdapter.partition()``.

    These exercise the genuine collectives (CP zigzag chunking via
    ``get_batch_on_this_cp_rank`` and the SP scatter via
    ``scatter_to_sequence_parallel_region``) against process groups built from a
    real ``HyperCommGrid``. They assert the actual per-rank output shapes *and*
    content rather than that a mock was invoked.

    Run with::

        WORLD_SIZE=8 python -m torch.distributed.run --nproc-per-node 8 -m pytest \
            tests/unit_tests/models/mimo/test_mimo_partition.py -m experimental \
            --experimental -k RealDistributed
    """

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _build_grid(tp_size, cp_size):
        """Build a real HyperCommGrid spanning all 8 ranks (remainder folded into dp)."""
        Utils.initialize_distributed()
        world_size = torch.distributed.get_world_size()
        assert world_size == 8, f"expected an 8-GPU world, got {world_size}"
        dp_size = world_size // (tp_size * cp_size)
        # Order tp-cp-...-dp so tp is the fastest-varying (contiguous SP shards).
        grid = HyperCommGrid([tp_size, cp_size, 1, 1, dp_size], ["tp", "cp", "ep", "pp", "dp"])
        return grid

    @staticmethod
    def _make_inputs(B, S, H):
        """Deterministic, rank-identical inputs so every rank shards the same tensor.

        Embeddings are sequence-first ``(S, B, H)`` and encode their (sequence, hidden)
        coordinates so a shard's content can be checked positionally; labels/loss_mask
        are ``(B, S)`` and encode the absolute sequence index.
        """
        torch.manual_seed(1234)
        seq = torch.arange(S, dtype=torch.float32)
        hid = torch.arange(H, dtype=torch.float32)
        # [S, B, H] where entry [s, b, h] = s * 1000 + h + b (unique per position).
        embeddings = (
            seq.view(S, 1, 1) * 1000.0
            + hid.view(1, 1, H)
            + torch.arange(B, dtype=torch.float32).view(1, B, 1)
        ).cuda()
        labels = torch.arange(S, dtype=torch.long).view(1, S).expand(B, S).contiguous().cuda()
        loss_mask = torch.arange(S, dtype=torch.float32).view(1, S).expand(B, S).contiguous().cuda()
        return embeddings, labels, loss_mask

    def test_sp_only_scatters_sequence_real(self):
        """SP-only: [S, B, H] -> [S/tp, B, H], contiguous sequence shard on this TP rank."""
        tp_size, cp_size = 8, 1
        B, S, H = 2, 64, 16
        grid = self._build_grid(tp_size, cp_size)
        tp_group = grid.create_pg("tp")

        cfg = PartitionConfig(
            use_cp=False,
            seq_parallel=True,
            tp_comm_overlap=False,
            max_seq_len=S,
            cp_group=None,
            tp_group=tp_group,
        )
        adapter = PartitionAdapter(cfg)
        embeddings, labels, loss_mask = self._make_inputs(B, S, H)

        cp_batch = adapter.partition(
            embeddings.clone(), None, None, labels.clone(), loss_mask.clone(), None
        )
        batch = cp_batch.get_batch()
        out_emb = batch["decoder_input"]
        out_labels = batch["labels"]
        out_loss_mask = batch["loss_mask"]

        tp_rank = tp_group.rank()
        shard = S // tp_size
        # SP scatter is a contiguous split along the sequence dim 0; no transpose for SP-only.
        assert out_emb.shape == (shard, B, H)
        expected = embeddings[tp_rank * shard : (tp_rank + 1) * shard]
        torch.testing.assert_close(out_emb, expected.contiguous())
        # Labels / loss_mask are NOT SP-scattered: full sequence comes back unchanged.
        assert out_labels.shape == (B, S)
        torch.testing.assert_close(out_labels, labels)
        assert out_loss_mask.shape == (B, S)
        torch.testing.assert_close(out_loss_mask, loss_mask)

    def test_cp_only_shards_sequence_real(self):
        """CP-only: embeddings -> [S/cp, B, H]; labels/loss_mask CP-sharded (zigzag), not scattered."""
        tp_size, cp_size = 1, 8
        B, S, H = 2, 64, 16
        grid = self._build_grid(tp_size, cp_size)
        cp_group = grid.create_pg("cp")

        cfg = PartitionConfig(
            use_cp=True,
            seq_parallel=False,
            tp_comm_overlap=False,
            max_seq_len=S,
            cp_group=cp_group,
            tp_group=None,
        )
        adapter = PartitionAdapter(cfg)
        embeddings, labels, loss_mask = self._make_inputs(B, S, H)

        cp_batch = adapter.partition(
            embeddings.clone(), None, None, labels.clone(), loss_mask.clone(), None
        )
        batch = cp_batch.get_batch()
        out_emb = batch["decoder_input"]
        out_labels = batch["labels"]
        out_loss_mask = batch["loss_mask"]

        cp_rank = cp_group.rank()
        shard = S // cp_size
        # Embeddings: transpose to batch-first for the zigzag, then back to [S/cp, B, H].
        assert out_emb.shape == (shard, B, H)
        emb_bshd = embeddings.transpose(0, 1)  # [S, B, H] -> [B, S, H]
        expected_emb = _expected_cp_zigzag_shard(emb_bshd, cp_size, cp_rank).transpose(0, 1)
        torch.testing.assert_close(out_emb, expected_emb.contiguous())
        # Labels / loss_mask: CP-sharded (zigzag) but NOT SP-scattered -> [B, S/cp].
        assert out_labels.shape == (B, shard)
        torch.testing.assert_close(out_labels, _expected_cp_zigzag_shard(labels, cp_size, cp_rank))
        assert out_loss_mask.shape == (B, shard)
        torch.testing.assert_close(
            out_loss_mask, _expected_cp_zigzag_shard(loss_mask, cp_size, cp_rank)
        )

    def test_cp_and_sp_combined_real(self):
        """CP+SP: embeddings -> [S/(cp*tp), B, H]; labels/loss_mask only CP-sharded [B, S/cp]."""
        tp_size, cp_size = 2, 2  # 2*2 = 4; remaining factor of 2 goes to dp (spans all 8 ranks).
        B, S, H = 2, 64, 16
        grid = self._build_grid(tp_size, cp_size)
        tp_group = grid.create_pg("tp")
        cp_group = grid.create_pg("cp")

        cfg = PartitionConfig(
            use_cp=True,
            seq_parallel=True,
            tp_comm_overlap=False,
            max_seq_len=S,
            cp_group=cp_group,
            tp_group=tp_group,
        )
        adapter = PartitionAdapter(cfg)
        embeddings, labels, loss_mask = self._make_inputs(B, S, H)

        cp_batch = adapter.partition(
            embeddings.clone(), None, None, labels.clone(), loss_mask.clone(), None
        )
        batch = cp_batch.get_batch()
        out_emb = batch["decoder_input"]
        out_labels = batch["labels"]
        out_loss_mask = batch["loss_mask"]

        cp_rank = cp_group.rank()
        tp_rank = tp_group.rank()
        cp_shard = S // cp_size
        final_shard = S // (cp_size * tp_size)

        # Embeddings: transpose to batch-first, CP zigzag, back to [S/cp, B, H], SP scatter dim 0.
        assert out_emb.shape == (final_shard, B, H)
        emb_bshd = embeddings.transpose(0, 1)  # [S, B, H] -> [B, S, H]
        cp_emb = _expected_cp_zigzag_shard(emb_bshd, cp_size, cp_rank).transpose(0, 1)
        expected_emb = cp_emb[tp_rank * final_shard : (tp_rank + 1) * final_shard]
        torch.testing.assert_close(out_emb, expected_emb.contiguous())

        # Labels / loss_mask: CP-sharded only (no SP scatter) -> [B, S/cp].
        assert out_labels.shape == (B, cp_shard)
        torch.testing.assert_close(out_labels, _expected_cp_zigzag_shard(labels, cp_size, cp_rank))
        assert out_loss_mask.shape == (B, cp_shard)
        torch.testing.assert_close(
            out_loss_mask, _expected_cp_zigzag_shard(loss_mask, cp_size, cp_rank)
        )
