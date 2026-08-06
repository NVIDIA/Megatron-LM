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

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.mimo.partition.utils import PartitionAdapter, PartitionConfig
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
        mp = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )
        with patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=1):
            cfg = PartitionConfig.from_mp_config(mp, max_seq_len=512, tp_group=mock_tp_group)
        assert cfg.seq_parallel is True
        assert cfg.tp_group is mock_tp_group

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
class TestPartitionAdapterShard:
    """Tests for PartitionAdapter.shard()."""

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
        """With neither CP nor SP active, shard() is a pure passthrough.

        Production never constructs a PartitionAdapter unless CP or SP is enabled.
        Embeddings are already sequence-first (S, B, H), so with no collectives the
        inputs are returned untouched (no transpose, no sharding).
        """
        cfg = self._make_cfg(use_cp=False, seq_parallel=False)
        adapter = PartitionAdapter(cfg)
        embeddings, labels, loss_mask = self._make_tensors(B=2, S=8, H=16)
        out = adapter.shard(embeddings, labels, loss_mask)
        assert out[0] is embeddings
        assert out[1] is labels
        assert out[2] is loss_mask
        assert out[3] is None

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
            adapter.shard(embeddings, labels, loss_mask)

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
            adapter.shard(embeddings, labels, loss_mask)

    def test_thd_format_skips_divisibility_check(self):
        """PackedSeqParams with qkv_format='thd' bypasses the divisibility assertion."""
        from megatron.core.packed_seq_params import PackedSeqParams

        mock_cp_group = MagicMock()
        cfg = self._make_cfg(use_cp=True, max_seq_len=7, cp_group=mock_cp_group)
        adapter = PartitionAdapter(cfg)
        embeddings = torch.rand(7, 2, 16)  # seq-first; len=7 not divisible by cp*2, THD skips check
        labels = torch.randint(0, 100, (2, 7))
        loss_mask = torch.ones(2, 7)
        packed_seq_params = MagicMock(spec=PackedSeqParams)
        packed_seq_params.qkv_format = 'thd'
        packed_seq_params.cu_seqlens_q_padded = torch.tensor([0, 4, 7], dtype=torch.int32)

        # THD path calls tex.thd_get_partitioned_indices — mock it to return first 4 indices
        fake_index = torch.arange(4, dtype=torch.int32)
        with (
            patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=2),
            patch('megatron.core.models.mimo.partition.utils.get_pg_rank', return_value=0),
            patch('megatron.core.models.mimo.partition.utils.tex') as mock_tex,
        ):
            mock_tex.thd_get_partitioned_indices.return_value = fake_index
            # Should NOT raise AssertionError about divisibility
            out = adapter.shard(embeddings, labels, loss_mask, packed_seq_params)
        assert out[0] is not None

    def test_none_embeddings_skips_shard_factor_check(self):
        """When embeddings is None, the divisibility check is skipped (non-first PP stage)."""
        mock_cp_group = MagicMock()
        cfg = self._make_cfg(use_cp=True, max_seq_len=7, cp_group=mock_cp_group)
        adapter = PartitionAdapter(cfg)
        labels = torch.randint(0, 100, (2, 7))
        loss_mask = torch.ones(2, 7)
        cp_sharded = {'labels': labels[:, :4], 'loss_mask': loss_mask[:, :4]}
        with (
            patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=2),
            patch(
                'megatron.core.models.mimo.partition.utils.get_batch_on_this_cp_rank',
                return_value=cp_sharded,
            ),
        ):
            out = adapter.shard(None, labels, loss_mask)
        assert out[0] is None
        assert out[1].shape == (2, 4)
        assert out[2].shape == (2, 4)


@pytest.mark.experimental
class TestPartitionAdapterApplyContextParallel:
    """Tests for PartitionAdapter._apply_context_parallel()."""

    def _make_cfg(self, use_cp=True, cp_group=None):
        return PartitionConfig(
            use_cp=use_cp,
            seq_parallel=False,
            tp_comm_overlap=False,
            max_seq_len=128,
            cp_group=cp_group,
        )

    def test_returns_unchanged_when_cp_disabled(self):
        cfg = self._make_cfg(use_cp=False)
        adapter = PartitionAdapter(cfg)
        embeddings = torch.rand(2, 8, 16)
        labels = torch.randint(0, 100, (2, 8))
        loss_mask = torch.ones(2, 8)
        out = adapter._apply_context_parallel(embeddings, labels, loss_mask, None)
        assert out[0] is embeddings
        assert out[1] is labels
        assert out[2] is loss_mask
        assert out[3] is None

    def test_sbhd_path_calls_get_batch_on_this_cp_rank(self):
        mock_cp_group = MagicMock()
        cfg = self._make_cfg(use_cp=True, cp_group=mock_cp_group)
        adapter = PartitionAdapter(cfg)
        embeddings = torch.rand(2, 8, 16)
        labels = torch.randint(0, 100, (2, 8))
        loss_mask = torch.ones(2, 8)
        sharded = {
            'embeddings': embeddings[:, :4, :],
            'labels': labels[:, :4],
            'loss_mask': loss_mask[:, :4],
        }
        with patch(
            'megatron.core.models.mimo.partition.utils.get_batch_on_this_cp_rank',
            return_value=sharded,
        ) as mock_fn:
            out = adapter._apply_context_parallel(embeddings, labels, loss_mask, None)
            mock_fn.assert_called_once()
        # _apply_context_parallel keeps batch-first [B, S/cp, H]; shard() transposes later.
        assert out[0].shape == (2, 4, 16)
        assert out[1].shape == (2, 4)

    def test_all_none_inputs_produces_none_outputs(self):
        mock_cp_group = MagicMock()
        cfg = self._make_cfg(use_cp=True, cp_group=mock_cp_group)
        adapter = PartitionAdapter(cfg)
        with patch(
            'megatron.core.models.mimo.partition.utils.get_batch_on_this_cp_rank', return_value={}
        ):
            out = adapter._apply_context_parallel(None, None, None, None)
        assert all(v is None for v in out[:3])

    def test_only_non_none_tensors_added_to_batch(self):
        """None tensors must not appear in the batch dict passed to get_batch_on_this_cp_rank."""
        mock_cp_group = MagicMock()
        cfg = self._make_cfg(use_cp=True, cp_group=mock_cp_group)
        adapter = PartitionAdapter(cfg)
        embeddings = torch.rand(2, 8, 16)
        sharded = {'embeddings': embeddings[:, :4, :]}
        captured = {}

        def mock_fn(batch, **kwargs):
            captured.update(batch)
            return sharded

        with patch(
            'megatron.core.models.mimo.partition.utils.get_batch_on_this_cp_rank',
            side_effect=mock_fn,
        ):
            out = adapter._apply_context_parallel(embeddings, None, None, None)

        assert 'embeddings' in captured
        assert 'labels' not in captured
        assert 'loss_mask' not in captured
        assert out[0] is not None
        assert out[1] is None

    def test_thd_path_raises_when_te_unavailable(self):
        """THD format must assert when Transformer Engine is not available."""
        from megatron.core.packed_seq_params import PackedSeqParams

        mock_cp_group = MagicMock()
        cfg = self._make_cfg(use_cp=True, cp_group=mock_cp_group)
        adapter = PartitionAdapter(cfg)
        embeddings = torch.rand(2, 5, 16)
        packed_seq_params = MagicMock(spec=PackedSeqParams)
        packed_seq_params.qkv_format = 'thd'
        with (
            patch('megatron.core.models.mimo.partition.utils._HAVE_TEX', False),
            pytest.raises(AssertionError, match="Transformer Engine"),
        ):
            adapter._apply_context_parallel(embeddings, None, None, packed_seq_params)


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
class TestPartitionAdapterShardRealDistributed:
    """Real 8-GPU tests for ``PartitionAdapter.shard()``.

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

        out_emb, out_labels, out_loss_mask, _ = adapter.shard(
            embeddings.clone(), labels.clone(), loss_mask.clone()
        )

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

        out_emb, out_labels, out_loss_mask, _ = adapter.shard(
            embeddings.clone(), labels.clone(), loss_mask.clone()
        )

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

        out_emb, out_labels, out_loss_mask, _ = adapter.shard(
            embeddings.clone(), labels.clone(), loss_mask.clone()
        )

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
