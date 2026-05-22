# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

'''
WORLD_SIZE=1 LOCAL_RANK=0 python -m torch.distributed.run \
    --nproc_per_node=1 -m pytest \
    tests/unit_tests/models/test_mimo_partition.py -v
'''

from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.models.mimo.partition.utils import PartitionAdapter, PartitionConfig
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.mark.experimental
class TestPartitionConfig:
    """Tests for PartitionConfig dataclass and factory method."""

    def test_is_partitioning_enabled_cp_only(self):
        cfg = PartitionConfig(
            seq_parallel=False, use_cp=True, tp_comm_overlap=False, max_seq_len=128
        )
        assert cfg.is_partitioning_enabled is True

    def test_is_partitioning_enabled_sp_only(self):
        cfg = PartitionConfig(
            seq_parallel=True, use_cp=False, tp_comm_overlap=False, max_seq_len=128
        )
        assert cfg.is_partitioning_enabled is True

    def test_is_partitioning_enabled_both(self):
        cfg = PartitionConfig(
            seq_parallel=True, use_cp=True, tp_comm_overlap=False, max_seq_len=128
        )
        assert cfg.is_partitioning_enabled is True

    def test_is_partitioning_enabled_neither(self):
        cfg = PartitionConfig(
            seq_parallel=False, use_cp=False, tp_comm_overlap=False, max_seq_len=128
        )
        assert cfg.is_partitioning_enabled is False

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
        embeddings = torch.rand(B, S, H)
        labels = torch.randint(0, 100, (B, S))
        loss_mask = torch.ones(B, S)
        attention_mask = torch.ones(B, S)
        return embeddings, labels, loss_mask, attention_mask

    def test_noop_when_both_disabled(self):
        """No sharding when neither CP nor SP is enabled — inputs returned as-is."""
        cfg = self._make_cfg(use_cp=False, seq_parallel=False)
        adapter = PartitionAdapter(cfg)
        embeddings, labels, loss_mask, attention_mask = self._make_tensors()
        out = adapter.shard(embeddings, labels, loss_mask, attention_mask)
        assert out[0] is embeddings
        assert out[1] is labels
        assert out[2] is loss_mask
        assert out[3] is attention_mask
        assert out[4] is None

    def test_cp_only_shards_sequence(self):
        mock_cp_group = MagicMock()
        cfg = self._make_cfg(use_cp=True, max_seq_len=8, cp_group=mock_cp_group)
        adapter = PartitionAdapter(cfg)
        embeddings, labels, loss_mask, attention_mask = self._make_tensors(B=2, S=8, H=16)
        sharded = {
            'embeddings': embeddings[:, :4, :],
            'labels': labels[:, :4],
            'loss_mask': loss_mask[:, :4],
            'attention_mask': attention_mask[:, :4],
        }
        with (
            patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=2),
            patch(
                'megatron.core.models.mimo.partition.utils.get_batch_on_this_cp_rank',
                return_value=sharded,
            ),
        ):
            out = adapter.shard(embeddings, labels, loss_mask, attention_mask)
        assert out[0].shape == (2, 4, 16)
        assert out[1].shape == (2, 4)

    def test_sp_only_scatters(self):
        mock_tp_group = MagicMock()
        cfg = self._make_cfg(seq_parallel=True, max_seq_len=8, tp_group=mock_tp_group)
        adapter = PartitionAdapter(cfg)
        # SP uses seq_dim=0: embeddings shape [S, B, H]
        embeddings = torch.rand(8, 2, 16)
        labels = torch.randint(0, 100, (2, 8))
        loss_mask = torch.ones(2, 8)
        attention_mask = torch.ones(2, 8)
        scattered = torch.rand(4, 2, 16)
        with (
            patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=2),
            patch(
                'megatron.core.models.mimo.partition.utils.tensor_parallel.scatter_to_sequence_parallel_region',
                return_value=scattered,
            ),
        ):
            out = adapter.shard(embeddings, labels, loss_mask, attention_mask)
        assert out[0].shape == (4, 2, 16)

    def test_cp_and_sp_combined(self):
        mock_cp_group = MagicMock()
        mock_tp_group = MagicMock()
        cfg = self._make_cfg(
            use_cp=True,
            seq_parallel=True,
            max_seq_len=16,
            cp_group=mock_cp_group,
            tp_group=mock_tp_group,
        )
        adapter = PartitionAdapter(cfg)
        # cp_size=2, tp_size=2 → shard_factor = 2*2*2 = 8; S=16 is divisible
        embeddings = torch.rand(2, 16, 16)
        labels = torch.randint(0, 100, (2, 16))
        loss_mask = torch.ones(2, 16)
        attention_mask = torch.ones(2, 16)
        cp_sharded = {
            'embeddings': embeddings[:, :8, :],
            'labels': labels[:, :8],
            'loss_mask': loss_mask[:, :8],
            'attention_mask': attention_mask[:, :8],
        }
        scattered = torch.rand(2, 4, 16)

        with (
            patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=2),
            patch(
                'megatron.core.models.mimo.partition.utils.get_batch_on_this_cp_rank',
                return_value=cp_sharded,
            ),
            patch(
                'megatron.core.models.mimo.partition.utils.tensor_parallel.scatter_to_sequence_parallel_region',
                return_value=scattered,
            ),
        ):
            out = adapter.shard(embeddings, labels, loss_mask, attention_mask)
        assert out[0].shape == (2, 4, 16)

    def test_seq_not_divisible_raises(self):
        mock_cp_group = MagicMock()
        cfg = self._make_cfg(use_cp=True, max_seq_len=7, cp_group=mock_cp_group)
        adapter = PartitionAdapter(cfg)
        embeddings = torch.rand(2, 7, 16)  # 7 % (2*2) != 0
        labels = torch.randint(0, 100, (2, 7))
        loss_mask = torch.ones(2, 7)
        attention_mask = torch.ones(2, 7)
        with (
            patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=2),
            pytest.raises(AssertionError, match="divisible"),
        ):
            adapter.shard(embeddings, labels, loss_mask, attention_mask)

    def test_tp_comm_overlap_seq_len_assertion(self):
        mock_tp_group = MagicMock()
        cfg = self._make_cfg(
            seq_parallel=True, tp_comm_overlap=True, max_seq_len=16, tp_group=mock_tp_group
        )
        adapter = PartitionAdapter(cfg)
        # S=8 but max_seq_len=16 → assertion fires
        embeddings = torch.rand(8, 2, 16)  # [S, B, H] for SP
        labels = torch.randint(0, 100, (2, 8))
        loss_mask = torch.ones(2, 8)
        attention_mask = torch.ones(2, 8)
        with (
            patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=2),
            pytest.raises(AssertionError, match="TP Comm overlap"),
        ):
            adapter.shard(embeddings, labels, loss_mask, attention_mask)

    def test_thd_format_skips_divisibility_check(self):
        """PackedSeqParams with qkv_format='thd' bypasses the divisibility assertion."""
        from megatron.core.packed_seq_params import PackedSeqParams

        mock_cp_group = MagicMock()
        cfg = self._make_cfg(use_cp=True, max_seq_len=7, cp_group=mock_cp_group)
        adapter = PartitionAdapter(cfg)
        embeddings = torch.rand(2, 7, 16)  # seq_len=7 not divisible by cp*2, but THD skips check
        labels = torch.randint(0, 100, (2, 7))
        loss_mask = torch.ones(2, 7)
        attention_mask = torch.ones(2, 7)
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
            out = adapter.shard(embeddings, labels, loss_mask, attention_mask, packed_seq_params)
        assert out[0] is not None

    def test_none_embeddings_skips_shard_factor_check(self):
        """When embeddings is None, the divisibility check is skipped."""
        mock_cp_group = MagicMock()
        cfg = self._make_cfg(use_cp=True, max_seq_len=7, cp_group=mock_cp_group)
        adapter = PartitionAdapter(cfg)
        labels = torch.randint(0, 100, (2, 7))
        loss_mask = torch.ones(2, 7)
        attention_mask = torch.ones(2, 7)
        cp_sharded = {
            'labels': labels[:, :4],
            'loss_mask': loss_mask[:, :4],
            'attention_mask': attention_mask[:, :4],
        }
        with (
            patch('megatron.core.models.mimo.partition.utils.get_pg_size', return_value=2),
            patch(
                'megatron.core.models.mimo.partition.utils.get_batch_on_this_cp_rank',
                return_value=cp_sharded,
            ),
        ):
            out = adapter.shard(None, labels, loss_mask, attention_mask)
        assert out[0] is None


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
        attention_mask = torch.ones(2, 8)
        out = adapter._apply_context_parallel(embeddings, labels, loss_mask, attention_mask, None)
        assert out[0] is embeddings
        assert out[1] is labels
        assert out[2] is loss_mask
        assert out[3] is attention_mask

    def test_sbhd_path_calls_get_batch_on_this_cp_rank(self):
        mock_cp_group = MagicMock()
        cfg = self._make_cfg(use_cp=True, cp_group=mock_cp_group)
        adapter = PartitionAdapter(cfg)
        embeddings = torch.rand(2, 8, 16)
        labels = torch.randint(0, 100, (2, 8))
        loss_mask = torch.ones(2, 8)
        attention_mask = torch.ones(2, 8)
        sharded = {
            'embeddings': embeddings[:, :4, :],
            'labels': labels[:, :4],
            'loss_mask': loss_mask[:, :4],
            'attention_mask': attention_mask[:, :4],
        }
        with patch(
            'megatron.core.models.mimo.partition.utils.get_batch_on_this_cp_rank',
            return_value=sharded,
        ) as mock_fn:
            out = adapter._apply_context_parallel(
                embeddings, labels, loss_mask, attention_mask, None
            )
            mock_fn.assert_called_once()
        assert out[0].shape == (2, 4, 16)
        assert out[1].shape == (2, 4)

    def test_all_none_inputs_produces_none_outputs(self):
        mock_cp_group = MagicMock()
        cfg = self._make_cfg(use_cp=True, cp_group=mock_cp_group)
        adapter = PartitionAdapter(cfg)
        with patch(
            'megatron.core.models.mimo.partition.utils.get_batch_on_this_cp_rank', return_value={}
        ):
            out = adapter._apply_context_parallel(None, None, None, None, None)
        assert all(v is None for v in out[:4])

    def test_only_non_none_tensors_added_to_batch(self):
        """None tensors must not appear in the batch dict passed to get_batch_on_this_cp_rank."""
        mock_cp_group = MagicMock()
        cfg = self._make_cfg(use_cp=True, cp_group=mock_cp_group)
        adapter = PartitionAdapter(cfg)
        embeddings = torch.rand(2, 8, 16)
        sharded = {'embeddings': embeddings[:, :4, :]}
        captured = {}

        def mock_fn(batch):
            captured.update(batch)
            return sharded

        with patch(
            'megatron.core.models.mimo.partition.utils.get_batch_on_this_cp_rank',
            side_effect=mock_fn,
        ):
            out = adapter._apply_context_parallel(embeddings, None, None, None, None)

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
            adapter._apply_context_parallel(embeddings, None, None, None, packed_seq_params)
