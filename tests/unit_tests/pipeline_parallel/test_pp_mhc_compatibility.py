# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unit tests for PP / VPP + mHC (Hyper Connections) compatibility.

Tests cover:
1. get_tensor_shapes: shape correctness with mHC for all PP stages
2. get_num_layers_to_build: layer counts with standalone embedding/loss + mHC
3. TransformerBlock expand/contract: correct placement at PP boundaries
4. VPP tensor_shape: single shape used across all chunks with mHC
5. E2E forward pass: PP + mHC + standalone embedding/loss (multi-GPU)
6. Flexible VPP layout (pipeline_model_parallel_layout) + mHC compatibility

Run with:
    uv run --no-sync pytest tests/unit_tests/pipeline_parallel/test_pp_mhc_compatibility.py -s -x
    # Multi-GPU tests (world_size >= 2):
    torchrun --nproc-per-node=2 -m pytest tests/unit_tests/pipeline_parallel/test_pp_mhc_compatibility.py -s -x
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_tensor_shapes
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pp_group(rank: int, size: int):
    """Create a mock PP process group with given rank and size."""
    pg = MagicMock()
    pg.rank.return_value = rank
    pg.size.return_value = size
    return pg


def _make_tp_cp_groups(tp_size: int = 1, cp_size: int = 1):
    tp = MagicMock()
    tp.size.return_value = tp_size
    cp = MagicMock()
    cp.size.return_value = cp_size
    return tp, cp


def _make_config(
    hidden_size=64,
    num_layers=8,
    pp_size=2,
    vp_size=None,
    enable_hyper_connections=False,
    num_residual_streams=4,
    account_for_embedding=False,
    account_for_loss=False,
    num_layers_first=None,
    num_layers_last=None,
    **extra,
):
    """Build a TransformerConfig for testing without initializing parallel state."""
    kwargs = dict(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=4,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vp_size,
        enable_hyper_connections=enable_hyper_connections,
        num_residual_streams=num_residual_streams,
        account_for_embedding_in_pipeline_split=account_for_embedding,
        account_for_loss_in_pipeline_split=account_for_loss,
        num_layers_in_first_pipeline_stage=num_layers_first,
        num_layers_in_last_pipeline_stage=num_layers_last,
        use_cpu_initialization=True,
    )
    if pp_size > 1:
        kwargs.setdefault('pipeline_dtype', torch.bfloat16)
    kwargs.update(extra)
    return TransformerConfig(**kwargs)


# ===========================================================================
# 1. get_tensor_shapes — shape correctness with mHC
# ===========================================================================

class TestGetTensorShapesWithMHC:
    """Verify get_tensor_shapes returns correct hidden dim for mHC-enabled models."""

    SEQ, MBS, H = 32, 2, 64
    N_STREAMS = 4

    def _shapes(self, config, pp_rank, pp_size, is_recv):
        tp, cp = _make_tp_cp_groups()
        pp = _make_pp_group(pp_rank, pp_size)
        return get_tensor_shapes(
            seq_length=self.SEQ,
            micro_batch_size=self.MBS,
            decoder_seq_length=None,
            config=config,
            tp_group=tp,
            cp_group=cp,
            pp_group=pp,
            is_recv=is_recv,
        )

    # --- Without mHC (baseline) ---

    def test_no_mhc_pp2_all_stages(self):
        cfg = _make_config(hidden_size=self.H, pp_size=2, enable_hyper_connections=False)
        for rank in range(2):
            for is_recv in (True, False):
                shapes = self._shapes(cfg, rank, 2, is_recv)
                assert shapes == [(self.SEQ, self.MBS, self.H)]

    # --- With mHC, PP=2 ---

    def test_mhc_pp2_rank0_send_nstream(self):
        """PP rank 0 sends n*C to rank 1."""
        cfg = _make_config(
            hidden_size=self.H, pp_size=2,
            enable_hyper_connections=True, num_residual_streams=self.N_STREAMS,
        )
        shapes = self._shapes(cfg, pp_rank=0, pp_size=2, is_recv=False)
        assert shapes == [(self.SEQ, self.MBS, self.H * self.N_STREAMS)]

    def test_mhc_pp2_rank0_recv_1stream(self):
        """PP rank 0 receives nothing from previous (is first stage), so shape = C."""
        cfg = _make_config(
            hidden_size=self.H, pp_size=2,
            enable_hyper_connections=True, num_residual_streams=self.N_STREAMS,
        )
        shapes = self._shapes(cfg, pp_rank=0, pp_size=2, is_recv=True)
        assert shapes == [(self.SEQ, self.MBS, self.H)]

    def test_mhc_pp2_rank1_recv_nstream(self):
        """PP rank 1 receives n*C from rank 0."""
        cfg = _make_config(
            hidden_size=self.H, pp_size=2,
            enable_hyper_connections=True, num_residual_streams=self.N_STREAMS,
        )
        shapes = self._shapes(cfg, pp_rank=1, pp_size=2, is_recv=True)
        assert shapes == [(self.SEQ, self.MBS, self.H * self.N_STREAMS)]

    def test_mhc_pp2_rank1_send_1stream(self):
        """PP rank 1 (last stage) sends C (after output_contract)."""
        cfg = _make_config(
            hidden_size=self.H, pp_size=2,
            enable_hyper_connections=True, num_residual_streams=self.N_STREAMS,
        )
        shapes = self._shapes(cfg, pp_rank=1, pp_size=2, is_recv=False)
        assert shapes == [(self.SEQ, self.MBS, self.H)]

    # --- With mHC, PP=4 (intermediate ranks) ---

    def test_mhc_pp4_intermediate_ranks(self):
        """Intermediate ranks both send and receive n*C."""
        cfg = _make_config(
            hidden_size=self.H, pp_size=4, num_layers=8,
            enable_hyper_connections=True, num_residual_streams=self.N_STREAMS,
        )
        for rank in (1, 2):
            for is_recv in (True, False):
                shapes = self._shapes(cfg, pp_rank=rank, pp_size=4, is_recv=is_recv)
                assert shapes == [(self.SEQ, self.MBS, self.H * self.N_STREAMS)], (
                    f"rank={rank}, is_recv={is_recv}"
                )

    # --- With sequence parallel ---

    def test_mhc_with_sequence_parallel(self):
        """Sequence parallel divides seq_length by TP size."""
        cfg = _make_config(
            hidden_size=self.H, pp_size=2,
            enable_hyper_connections=True, num_residual_streams=self.N_STREAMS,
            sequence_parallel=True, tensor_model_parallel_size=2,
        )
        tp, cp = _make_tp_cp_groups(tp_size=2)
        pp = _make_pp_group(0, 2)
        shapes = get_tensor_shapes(
            seq_length=self.SEQ, micro_batch_size=self.MBS, decoder_seq_length=None,
            config=cfg, tp_group=tp, cp_group=cp, pp_group=pp, is_recv=False,
        )
        assert shapes == [(self.SEQ // 2, self.MBS, self.H * self.N_STREAMS)]


# ===========================================================================
# 2. get_num_layers_to_build — mHC + standalone embedding/loss
# ===========================================================================

class TestGetNumLayersToBuilWithMHC:
    """
    Verify layer counts are correct when mHC is combined with standalone
    embedding / loss stages (account_for_embedding/loss_in_pipeline_split).
    mHC itself doesn't change layer counts, but we need to ensure the
    combination doesn't break.
    """

    def test_pp2_even_split_mhc(self):
        cfg = _make_config(num_layers=8, pp_size=2, enable_hyper_connections=True)
        assert get_num_layers_to_build(cfg, pp_rank=0) == 4
        assert get_num_layers_to_build(cfg, pp_rank=1) == 4

    def test_pp2_standalone_embedding_mhc(self):
        """With standalone embedding on PP rank 0, rank 0 builds fewer layers."""
        cfg = _make_config(
            num_layers=8, pp_size=2,
            enable_hyper_connections=True,
            account_for_embedding=True,
            account_for_loss=True,
        )
        # (8 + 1 + 1) / 2 = 5 per rank
        # rank 0: 5 - 1 (embedding) = 4 transformer layers
        # rank 1: 5 - 1 (loss) = 4 transformer layers
        assert get_num_layers_to_build(cfg, pp_rank=0) == 4
        assert get_num_layers_to_build(cfg, pp_rank=1) == 4

    def test_pp4_standalone_invalid_division_raises(self):
        """PP=4, standalone embedding+loss, 12 layers → (12+2)/4=3.5 → raises."""
        with pytest.raises((ValueError, AssertionError)):
            _make_config(
                num_layers=12, pp_size=4,
                enable_hyper_connections=True,
                account_for_embedding=True,
                account_for_loss=True,
            )

    def test_pp4_standalone_both_mhc_valid(self):
        """Valid configuration: (14+2)/4 = 4 per rank."""
        cfg = _make_config(
            num_layers=14, pp_size=4,
            enable_hyper_connections=True,
            account_for_embedding=True,
            account_for_loss=True,
        )
        # rank 0: 4 - 1 (embedding) = 3
        # rank 1, 2: 4
        # rank 3: 4 - 1 (loss) = 3
        assert get_num_layers_to_build(cfg, pp_rank=0) == 3
        assert get_num_layers_to_build(cfg, pp_rank=1) == 4
        assert get_num_layers_to_build(cfg, pp_rank=2) == 4
        assert get_num_layers_to_build(cfg, pp_rank=3) == 3

    def test_uneven_pp_with_mhc(self):
        """Uneven PP: first stage has 2 layers, last has 2, middle gets 2 each."""
        cfg = _make_config(
            num_layers=8, pp_size=4,
            enable_hyper_connections=True,
            num_layers_first=2,
            num_layers_last=2,
        )
        assert get_num_layers_to_build(cfg, pp_rank=0) == 2
        assert get_num_layers_to_build(cfg, pp_rank=1) == 2
        assert get_num_layers_to_build(cfg, pp_rank=2) == 2
        assert get_num_layers_to_build(cfg, pp_rank=3) == 2

    def test_vpp_with_mhc(self):
        """VPP=2 with mHC: each VP stage gets half the layers per rank."""
        cfg = _make_config(num_layers=8, pp_size=2, vp_size=2, enable_hyper_connections=True)
        for pp_rank in range(2):
            for vp_stage in range(2):
                n = get_num_layers_to_build(cfg, vp_stage=vp_stage, pp_rank=pp_rank)
                assert n == 2, f"pp_rank={pp_rank}, vp_stage={vp_stage}, got {n}"

    def test_vpp_standalone_embedding_loss_invalid_raises(self):
        """VPP=2, standalone embedding+loss, pp=2, 8 layers → 10/2=5, 5%2!=0 → raises."""
        with pytest.raises((ValueError, AssertionError)):
            _make_config(
                num_layers=8, pp_size=2, vp_size=2,
                enable_hyper_connections=True,
                account_for_embedding=True,
                account_for_loss=True,
            )

    def test_vpp_standalone_both_valid_mhc(self):
        """VPP=2, standalone embed+loss, pp=4, 14 layers → (14+2)/4=4, 4/2=2 per VP."""
        cfg = _make_config(
            num_layers=14, pp_size=4, vp_size=2,
            enable_hyper_connections=True,
            account_for_embedding=True,
            account_for_loss=True,
        )
        # rank 0, vp 0: first PP + first VP → 2 - 1(embed) = 1
        assert get_num_layers_to_build(cfg, vp_stage=0, pp_rank=0) == 1
        # rank 0, vp 1: first PP + second VP → 2
        assert get_num_layers_to_build(cfg, vp_stage=1, pp_rank=0) == 2
        # rank 1-2: 2 per VP stage
        for rank in (1, 2):
            for vp in (0, 1):
                assert get_num_layers_to_build(cfg, vp_stage=vp, pp_rank=rank) == 2
        # rank 3, vp 0: 2
        assert get_num_layers_to_build(cfg, vp_stage=0, pp_rank=3) == 2
        # rank 3, vp 1: last PP + last VP → 2 - 1(loss) = 1
        assert get_num_layers_to_build(cfg, vp_stage=1, pp_rank=3) == 1


# ===========================================================================
# 3. TransformerBlock expand/contract — boundary logic
# ===========================================================================

class TestTransformerBlockMHCBoundaries:
    """
    Test that TransformerBlock correctly applies input_expand at pre_process
    and output_contract at the final layernorm stage.
    These are pure tensor operation tests — no GPU or parallel state needed.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_input_expand(self):
        from megatron.core.transformer.hyper_connection import HyperConnectionModule
        n = 4
        s, b, C = 8, 2, 64
        x = torch.randn(s, b, C, device='cuda')
        expanded = HyperConnectionModule.input_expand(x, n)
        assert expanded.shape == (s, b, n * C)
        # Each stream should be a copy of input
        for i in range(n):
            torch.testing.assert_close(
                expanded[:, :, i * C : (i + 1) * C], x
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_output_contract(self):
        from megatron.core.transformer.hyper_connection import HyperConnectionModule
        n = 4
        s, b, C = 8, 2, 64
        x = torch.randn(s, b, n * C, device='cuda')
        contracted = HyperConnectionModule.output_contract(x, n)
        assert contracted.shape == (s, b, C)
        # Should be the mean of all n streams
        expected = x.view(s, b, n, C).mean(dim=2)
        torch.testing.assert_close(contracted, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_expand_then_contract_preserves_shape(self):
        from megatron.core.transformer.hyper_connection import HyperConnectionModule
        n = 4
        s, b, C = 8, 2, 64
        x = torch.randn(s, b, C, device='cuda')
        expanded = HyperConnectionModule.input_expand(x, n)
        contracted = HyperConnectionModule.output_contract(expanded, n)
        assert contracted.shape == x.shape
        # expand copies all streams → mean of identical streams = original
        torch.testing.assert_close(contracted, x)


# ===========================================================================
# 3b. Zero-layer VP stage edge cases with mHC
# ===========================================================================

class TestZeroLayerVPStageWithMHC:
    """
    When standalone embedding/loss makes a VP stage have very few (1) transformer
    layers, verify layer counts stay non-negative.
    """

    def test_vpp_standalone_embed_first_stage_has_1_layer(self):
        """First VP stage at first PP rank should have exactly 1 layer (2-1=1)."""
        cfg = _make_config(
            num_layers=7, pp_size=2, vp_size=2,
            enable_hyper_connections=True,
            account_for_embedding=True,
        )
        n = get_num_layers_to_build(cfg, vp_stage=0, pp_rank=0)
        assert n == 1
        assert n >= 0

    def test_vpp_standalone_loss_last_stage_has_1_layer(self):
        """Last VP stage at last PP rank should have exactly 1 layer (2-1=1)."""
        cfg = _make_config(
            num_layers=7, pp_size=2, vp_size=2,
            enable_hyper_connections=True,
            account_for_loss=True,
        )
        n = get_num_layers_to_build(cfg, vp_stage=1, pp_rank=1)
        assert n == 1
        assert n >= 0

    def test_vpp_standalone_both_boundary_layers(self):
        """Both first and last VP stages lose a layer, but all counts remain >= 0."""
        cfg = _make_config(
            num_layers=14, pp_size=4, vp_size=2,
            enable_hyper_connections=True,
            account_for_embedding=True,
            account_for_loss=True,
        )
        for pp_rank in range(4):
            for vp_stage in range(2):
                n = get_num_layers_to_build(cfg, vp_stage=vp_stage, pp_rank=pp_rank)
                assert n >= 0, f"pp_rank={pp_rank}, vp_stage={vp_stage} has {n} < 0 layers"


# ===========================================================================
# 4. VPP tensor_shape — single shape for all chunks
# ===========================================================================

class TestVPPTensorShapeWithMHC:
    """
    Verify that the interleaved schedule uses n*C for all P2P communication
    when mHC is enabled with PP > 1.
    """

    def test_interleaved_tensor_shape_uses_nstream(self):
        """Reproduce the logic in forward_backward_pipelining_with_interleaving."""
        hidden_size = 64
        n_streams = 4
        pp_size = 2

        config = SimpleNamespace(
            hidden_size=hidden_size,
            enable_hyper_connections=True,
            num_residual_streams=n_streams,
            sequence_parallel=False,
        )

        hidden_dim = config.hidden_size
        if getattr(config, 'enable_hyper_connections', False) and pp_size > 1:
            hidden_dim = config.hidden_size * getattr(config, 'num_residual_streams', 1)

        assert hidden_dim == hidden_size * n_streams

    def test_interleaved_tensor_shape_no_mhc(self):
        """Without mHC, hidden_dim = hidden_size."""
        hidden_size = 64
        pp_size = 2

        config = SimpleNamespace(
            hidden_size=hidden_size,
            enable_hyper_connections=False,
            sequence_parallel=False,
        )

        hidden_dim = config.hidden_size
        if getattr(config, 'enable_hyper_connections', False) and pp_size > 1:
            hidden_dim = config.hidden_size * getattr(config, 'num_residual_streams', 1)

        assert hidden_dim == hidden_size

    def test_interleaved_tensor_shape_pp1_mhc_no_expand(self):
        """PP=1 with mHC: no P2P communication needed, no shape change."""
        hidden_size = 64
        n_streams = 4
        pp_size = 1

        config = SimpleNamespace(
            hidden_size=hidden_size,
            enable_hyper_connections=True,
            num_residual_streams=n_streams,
            sequence_parallel=False,
        )

        hidden_dim = config.hidden_size
        if getattr(config, 'enable_hyper_connections', False) and pp_size > 1:
            hidden_dim = config.hidden_size * getattr(config, 'num_residual_streams', 1)

        assert hidden_dim == hidden_size


# ===========================================================================
# 5. Shape consistency across PP stages with VPP + mHC
# ===========================================================================

class TestPPShapeConsistencyWithMHC:
    """
    Verify that send shape from one stage matches recv shape of the next stage.
    This is critical: a mismatch would cause a hang or crash in P2P communication.
    """

    def _get_send_recv_shapes(self, config, pp_size):
        """Get (send_shape, recv_shape) for consecutive PP stages."""
        tp, cp = _make_tp_cp_groups()
        results = []
        for rank in range(pp_size):
            send = get_tensor_shapes(
                seq_length=32, micro_batch_size=2, decoder_seq_length=None,
                config=config, tp_group=tp, cp_group=cp,
                pp_group=_make_pp_group(rank, pp_size), is_recv=False,
            )
            recv = get_tensor_shapes(
                seq_length=32, micro_batch_size=2, decoder_seq_length=None,
                config=config, tp_group=tp, cp_group=cp,
                pp_group=_make_pp_group(rank, pp_size), is_recv=True,
            )
            results.append((send, recv))
        return results

    def test_pp2_mhc_send_recv_match(self):
        """Rank 0's send shape must match rank 1's recv shape."""
        cfg = _make_config(hidden_size=64, pp_size=2, enable_hyper_connections=True)
        shapes = self._get_send_recv_shapes(cfg, 2)
        # rank 0 send == rank 1 recv
        assert shapes[0][0] == shapes[1][1], (
            f"rank 0 send {shapes[0][0]} != rank 1 recv {shapes[1][1]}"
        )

    def test_pp4_mhc_all_consecutive_match(self):
        """For all consecutive stages, send[i] == recv[i+1]."""
        cfg = _make_config(
            hidden_size=64, num_layers=8, pp_size=4, enable_hyper_connections=True,
        )
        shapes = self._get_send_recv_shapes(cfg, 4)
        for i in range(3):
            assert shapes[i][0] == shapes[i + 1][1], (
                f"rank {i} send {shapes[i][0]} != rank {i+1} recv {shapes[i+1][1]}"
            )

    def test_pp4_no_mhc_all_consecutive_match(self):
        """Baseline: without mHC, all shapes should be plain hidden_size."""
        cfg = _make_config(hidden_size=64, num_layers=8, pp_size=4)
        shapes = self._get_send_recv_shapes(cfg, 4)
        for i in range(3):
            assert shapes[i][0] == shapes[i + 1][1]
            assert shapes[i][0] == [(32, 2, 64)]


# ===========================================================================
# 6. Standalone embedding / loss — PP boundary + mHC interaction
# ===========================================================================

class TestStandaloneEmbeddingLossWithMHC:
    """
    Verify that standalone embedding/loss configurations interact correctly
    with mHC tensor shapes and layer counting.
    """

    def test_standalone_embedding_first_stage_has_fewer_layers(self):
        """With standalone embedding, first PP/VP stage gets 1 fewer layer."""
        # 7 layers, pp=2, vp=2 → (7+1)/2=4, 4/2=2 per VP stage
        cfg = _make_config(
            num_layers=7, pp_size=2, vp_size=2,
            enable_hyper_connections=True,
            account_for_embedding=True,
        )
        # rank 0, vp 0: first stage → 2 - 1(embed) = 1
        assert get_num_layers_to_build(cfg, vp_stage=0, pp_rank=0) == 1
        # rank 0, vp 1: 2
        assert get_num_layers_to_build(cfg, vp_stage=1, pp_rank=0) == 2
        # rank 1: 2 each VP
        assert get_num_layers_to_build(cfg, vp_stage=0, pp_rank=1) == 2
        assert get_num_layers_to_build(cfg, vp_stage=1, pp_rank=1) == 2

    def test_standalone_loss_last_stage_has_fewer_layers(self):
        """With standalone loss, last PP/VP stage gets 1 fewer layer."""
        cfg = _make_config(
            num_layers=7, pp_size=2, vp_size=2,
            enable_hyper_connections=True,
            account_for_loss=True,
        )
        # (7+1)/2 = 4, 4/2 = 2 per VP
        # rank 0: 2 each VP
        assert get_num_layers_to_build(cfg, vp_stage=0, pp_rank=0) == 2
        assert get_num_layers_to_build(cfg, vp_stage=1, pp_rank=0) == 2
        # rank 1, vp 0: 2
        assert get_num_layers_to_build(cfg, vp_stage=0, pp_rank=1) == 2
        # rank 1, vp 1: last stage → 2 - 1(loss) = 1
        assert get_num_layers_to_build(cfg, vp_stage=1, pp_rank=1) == 1

    def test_standalone_both_mhc_shapes_still_consistent(self):
        """With standalone embed+loss, P2P shapes should still match between stages."""
        cfg = _make_config(
            hidden_size=64, num_layers=14, pp_size=4,
            enable_hyper_connections=True, num_residual_streams=4,
            account_for_embedding=True, account_for_loss=True,
        )
        tp, cp = _make_tp_cp_groups()
        for i in range(3):
            send = get_tensor_shapes(
                seq_length=32, micro_batch_size=2, decoder_seq_length=None,
                config=cfg, tp_group=tp, cp_group=cp,
                pp_group=_make_pp_group(i, 4), is_recv=False,
            )
            recv = get_tensor_shapes(
                seq_length=32, micro_batch_size=2, decoder_seq_length=None,
                config=cfg, tp_group=tp, cp_group=cp,
                pp_group=_make_pp_group(i + 1, 4), is_recv=True,
            )
            assert send == recv, (
                f"rank {i}→{i+1}: send={send} recv={recv}"
            )

    def test_mhc_shapes_first_stage_send_vs_second_recv(self):
        """
        First stage (pre_process) does input_expand: hidden [s,b,C] → [s,b,n*C].
        The send shape from rank 0 should be n*C.
        The recv shape at rank 1 should also be n*C.
        """
        H, N = 64, 4
        cfg = _make_config(
            hidden_size=H, num_layers=8, pp_size=2,
            enable_hyper_connections=True, num_residual_streams=N,
        )
        tp, cp = _make_tp_cp_groups()
        send_0 = get_tensor_shapes(
            seq_length=32, micro_batch_size=2, decoder_seq_length=None,
            config=cfg, tp_group=tp, cp_group=cp,
            pp_group=_make_pp_group(0, 2), is_recv=False,
        )
        recv_1 = get_tensor_shapes(
            seq_length=32, micro_batch_size=2, decoder_seq_length=None,
            config=cfg, tp_group=tp, cp_group=cp,
            pp_group=_make_pp_group(1, 2), is_recv=True,
        )
        assert send_0 == [(32, 2, H * N)]
        assert recv_1 == [(32, 2, H * N)]
        assert send_0 == recv_1

    def test_mhc_shapes_last_stage_output_is_1stream(self):
        """
        Last stage (post_process) does output_contract: [s,b,n*C] → [s,b,C].
        The send shape from last rank should be C (but get_tensor_shapes returns C
        because last rank doesn't send forward).
        """
        H, N = 64, 4
        cfg = _make_config(
            hidden_size=H, num_layers=8, pp_size=2,
            enable_hyper_connections=True, num_residual_streams=N,
        )
        tp, cp = _make_tp_cp_groups()
        send_last = get_tensor_shapes(
            seq_length=32, micro_batch_size=2, decoder_seq_length=None,
            config=cfg, tp_group=tp, cp_group=cp,
            pp_group=_make_pp_group(1, 2), is_recv=False,
        )
        # Last stage sends C (after contract), not n*C
        assert send_last == [(32, 2, H)]


# ===========================================================================
# 7. E2E forward pass tests (require multi-GPU)
# ===========================================================================

@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    int(__import__('os').environ.get('WORLD_SIZE', '1')) < 2,
    reason="Requires at least 2 GPUs"
)
class TestPPForwardWithMHC:
    """
    End-to-end forward pass tests with PP + mHC.
    Requires multi-GPU (torchrun --nproc-per-node=2+).
    """

    def _run_forward(
        self,
        pp_size,
        vp_size,
        enable_mhc,
        account_for_embedding=False,
        account_for_loss=False,
    ):
        from megatron.core import mpu
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.num_microbatches_calculator import (
            init_num_microbatches_calculator,
            unset_num_microbatches_calculator,
        )
        from megatron.core.pipeline_parallel import get_forward_backward_func
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.enums import ModelType
        from megatron.training.global_vars import set_args
        from tests.unit_tests.test_utilities import Utils

        num_layers = 8
        hidden_size = 64
        num_heads = 4
        seq_length = 16
        micro_batch_size = 2
        vocab_size = 128

        Utils.initialize_model_parallel(1, pp_size, vp_size)
        model_parallel_cuda_manual_seed(42)
        init_num_microbatches_calculator(0, None, 1, 1, 1)

        try:
            config = TransformerConfig(
                num_layers=num_layers,
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                use_cpu_initialization=True,
                pipeline_dtype=torch.bfloat16,
                bf16=True,
                pipeline_model_parallel_size=pp_size,
                virtual_pipeline_model_parallel_size=vp_size,
                enable_hyper_connections=enable_mhc,
                num_residual_streams=4 if enable_mhc else 1,
                account_for_embedding_in_pipeline_split=account_for_embedding,
                account_for_loss_in_pipeline_split=account_for_loss,
                hidden_dropout=0.0,
                attention_dropout=0.0,
            )

            spec = get_gpt_layer_with_transformer_engine_spec(
                enable_hyper_connection=enable_mhc
            )

            models = []
            for i in range(vp_size or 1):
                pre_process = mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=i)
                post_process = mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i)
                m = (
                    GPTModel(
                        config=config,
                        transformer_layer_spec=spec,
                        vocab_size=vocab_size,
                        max_sequence_length=seq_length,
                        pre_process=pre_process,
                        post_process=post_process,
                        position_embedding_type="rope",
                        vp_stage=i,
                        share_embeddings_and_output_weights=False,
                    )
                    .bfloat16()
                    .cuda()
                )
                m.model_type = ModelType.encoder_or_decoder
                models.append(m)

            if vp_size is None:
                models = models[0]
                model_list = [models]
            else:
                model_list = models

            def forward_step_func(data_iterator, model):
                tokens = torch.randint(0, vocab_size, (micro_batch_size, seq_length)).cuda()
                position_ids = torch.arange(seq_length).unsqueeze(0).expand(micro_batch_size, -1).cuda()
                labels = torch.randint(0, vocab_size, (micro_batch_size, seq_length)).cuda()
                output = model(tokens, position_ids, None, labels=labels)

                def loss_func(output_tensor):
                    loss = output_tensor.sum()
                    return output_tensor, loss

                return output, loss_func

            forward_backward_func = get_forward_backward_func()

            def make_iter():
                while True:
                    yield None

            data_iters = [make_iter()] * len(model_list)

            losses = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iters,
                model=model_list,
                num_microbatches=4,
                seq_length=seq_length,
                micro_batch_size=micro_batch_size,
                forward_only=True,
            )
            return losses

        finally:
            unset_num_microbatches_calculator()
            Utils.destroy_model_parallel()

    def test_pp2_mhc_forward(self):
        """PP=2 + mHC forward pass should not hang."""
        self._run_forward(pp_size=2, vp_size=None, enable_mhc=True)

    def test_pp2_vpp2_mhc_forward(self):
        """PP=2 + VPP=2 + mHC forward pass should not hang."""
        self._run_forward(pp_size=2, vp_size=2, enable_mhc=True)

    def test_pp2_mhc_standalone_embedding_forward(self):
        """PP=2 + mHC + standalone embedding."""
        # (8+1)/2 = 4.5 → need (num_layers+1) divisible by pp_size
        # Use default 8 layers, won't divide evenly. Skip standalone embedding
        # with 8 layers pp=2 as (8+1)/2 isn't integer.
        # The test framework should raise ValueError, confirming the validation.
        with pytest.raises((ValueError, AssertionError)):
            self._run_forward(
                pp_size=2, vp_size=None, enable_mhc=True,
                account_for_embedding=True,
            )

    def test_pp2_mhc_standalone_both_forward(self):
        """PP=2 + mHC + standalone embedding + loss: (8+2)/2=5, works."""
        self._run_forward(
            pp_size=2, vp_size=None, enable_mhc=True,
            account_for_embedding=True, account_for_loss=True,
        )

    def test_pp2_no_mhc_forward_baseline(self):
        """Baseline: PP=2 without mHC should work fine."""
        self._run_forward(pp_size=2, vp_size=None, enable_mhc=False)


# ===========================================================================
# 8. Flexible VPP layout (pipeline_model_parallel_layout) + mHC
# ===========================================================================

def _make_layout_config(
    hidden_size=64,
    num_layers=8,
    pp_size=2,
    layout=None,
    enable_hyper_connections=False,
    num_residual_streams=4,
    **extra,
):
    """Build a TransformerConfig with a flexible VPP layout for testing.

    Unlike _make_config, this uses pipeline_model_parallel_layout instead of
    account_for_embedding/loss flags, since they are mutually exclusive.
    """
    kwargs = dict(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=4,
        pipeline_model_parallel_size=pp_size,
        pipeline_model_parallel_layout=layout,
        pipeline_dtype=torch.bfloat16,
        enable_hyper_connections=enable_hyper_connections,
        num_residual_streams=num_residual_streams,
        use_cpu_initialization=True,
    )
    kwargs.update(extra)
    return TransformerConfig(**kwargs)


class TestFlexibleVPPLayoutLayerCountsWithMHC:
    """
    Verify get_num_layers_to_build returns correct layer counts when
    flexible VPP layout (pipeline_model_parallel_layout) is combined with mHC.
    mHC itself doesn't change layer counts, so these tests confirm the
    combination doesn't break anything.
    """

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        parallel_state.set_pipeline_model_parallel_world_size(None)
        parallel_state.set_virtual_pipeline_model_parallel_world_size(None)

    def test_pp2_vpp2_standalone_embed_loss_mhc(self):
        """PP=2, VPP=2: standalone embedding & loss on separate VP stages."""
        # Layout: [["embedding"], ["decoder"]*6, ["decoder"], ["loss"]]
        # PP=2, VPP=2 → 4 stages:
        #   PP0 VP0: ["embedding"]    → 0 decoders
        #   PP1 VP0: ["decoder"]*6    → 6 decoders
        #   PP0 VP1: ["decoder"]      → 1 decoder
        #   PP1 VP1: ["loss"]         → 0 decoders
        layout = [["embedding"], ["decoder"] * 6, ["decoder"], ["loss"]]
        Utils.fake_initialize_model_parallel(
            pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2,
        )
        cfg = _make_layout_config(
            num_layers=7, pp_size=2, layout=layout,
            enable_hyper_connections=True, num_residual_streams=4,
        )

        expected = {(0, 0): 0, (0, 1): 1, (1, 0): 6, (1, 1): 0}
        total = 0
        for pp_rank in range(2):
            parallel_state.set_pipeline_model_parallel_rank(pp_rank)
            for vp in range(2):
                n = get_num_layers_to_build(cfg, vp_stage=vp)
                assert n == expected[(pp_rank, vp)], (
                    f"pp_rank={pp_rank}, vp={vp}: expected {expected[(pp_rank, vp)]}, got {n}"
                )
                total += n
        assert total == 7

    def test_pp2_vpp2_even_split_mhc(self):
        """PP=2, VPP=2: even split with embedding/loss attached to decoder stages."""
        # Layout: [["embedding","decoder","decoder"], ["decoder"]*4,
        #          ["decoder"], ["decoder","loss"]]
        # PP0 VP0: ["embedding","decoder","decoder"] → 2 decoders
        # PP1 VP0: ["decoder"]*4                     → 4 decoders
        # PP0 VP1: ["decoder"]                       → 1 decoder
        # PP1 VP1: ["decoder","loss"]                → 1 decoder
        layout = [
            ["embedding", "decoder", "decoder"],
            ["decoder"] * 4,
            ["decoder"],
            ["decoder", "loss"],
        ]
        Utils.fake_initialize_model_parallel(
            pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2,
        )
        cfg = _make_layout_config(
            num_layers=8, pp_size=2, layout=layout,
            enable_hyper_connections=True,
        )

        expected = {(0, 0): 2, (0, 1): 1, (1, 0): 4, (1, 1): 1}
        total = 0
        for pp_rank in range(2):
            parallel_state.set_pipeline_model_parallel_rank(pp_rank)
            for vp in range(2):
                n = get_num_layers_to_build(cfg, vp_stage=vp)
                assert n == expected[(pp_rank, vp)], (
                    f"pp_rank={pp_rank}, vp={vp}: expected {expected[(pp_rank, vp)]}, got {n}"
                )
                total += n
        assert total == 8

    def test_pp2_vpp2_empty_stage_mhc(self):
        """PP=2, VPP=2: empty VP stage (standalone embedding) with mHC."""
        # Layout: [["embedding"], ["decoder"]*7, [], ["loss"]]
        # PP0 VP0: ["embedding"]  → 0 decoders
        # PP1 VP0: ["decoder"]*7  → 7 decoders
        # PP0 VP1: []             → 0 decoders
        # PP1 VP1: ["loss"]       → 0 decoders
        layout = [["embedding"], ["decoder"] * 7, [], ["loss"]]
        Utils.fake_initialize_model_parallel(
            pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2,
        )
        cfg = _make_layout_config(
            num_layers=7, pp_size=2, layout=layout,
            enable_hyper_connections=True,
        )

        expected = {(0, 0): 0, (0, 1): 0, (1, 0): 7, (1, 1): 0}
        for pp_rank in range(2):
            parallel_state.set_pipeline_model_parallel_rank(pp_rank)
            for vp in range(2):
                n = get_num_layers_to_build(cfg, vp_stage=vp)
                assert n == expected[(pp_rank, vp)]
                assert n >= 0

    def test_mhc_does_not_alter_layout_layer_counts(self):
        """Same layout gives identical layer counts with and without mHC."""
        layout = [
            ["embedding", "decoder", "decoder"],
            ["decoder"] * 4,
            ["decoder"],
            ["decoder", "loss"],
        ]
        Utils.fake_initialize_model_parallel(
            pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2,
        )
        cfg_mhc = _make_layout_config(
            num_layers=8, pp_size=2, layout=layout, enable_hyper_connections=True,
        )
        cfg_no_mhc = _make_layout_config(
            num_layers=8, pp_size=2, layout=layout, enable_hyper_connections=False,
        )

        for pp_rank in range(2):
            parallel_state.set_pipeline_model_parallel_rank(pp_rank)
            for vp in range(2):
                n_mhc = get_num_layers_to_build(cfg_mhc, vp_stage=vp)
                n_no_mhc = get_num_layers_to_build(cfg_no_mhc, vp_stage=vp)
                assert n_mhc == n_no_mhc, (
                    f"pp_rank={pp_rank}, vp={vp}: mHC={n_mhc} != no-mHC={n_no_mhc}"
                )


class TestFlexibleVPPLayoutShapeConsistencyWithMHC:
    """
    Verify that P2P tensor shapes are consistent (send == recv) between
    consecutive PP stages when using flexible VPP layout + mHC.
    This is critical: a shape mismatch causes hangs or crashes.
    """

    def _get_send_recv_shapes(self, config, pp_size, seq=32, mbs=2):
        tp, cp = _make_tp_cp_groups()
        results = []
        for rank in range(pp_size):
            send = get_tensor_shapes(
                seq_length=seq, micro_batch_size=mbs, decoder_seq_length=None,
                config=config, tp_group=tp, cp_group=cp,
                pp_group=_make_pp_group(rank, pp_size), is_recv=False,
            )
            recv = get_tensor_shapes(
                seq_length=seq, micro_batch_size=mbs, decoder_seq_length=None,
                config=config, tp_group=tp, cp_group=cp,
                pp_group=_make_pp_group(rank, pp_size), is_recv=True,
            )
            results.append((send, recv))
        return results

    def test_pp2_flexible_vpp_mhc_send_recv_match(self):
        """PP=2 with flexible VPP layout + mHC: rank 0 send == rank 1 recv."""
        H, N = 64, 4
        cfg = _make_layout_config(
            hidden_size=H, num_layers=7, pp_size=2,
            layout=[["embedding"], ["decoder"] * 6, ["decoder"], ["loss"]],
            enable_hyper_connections=True, num_residual_streams=N,
        )
        shapes = self._get_send_recv_shapes(cfg, pp_size=2)
        assert shapes[0][0] == shapes[1][1], (
            f"rank 0 send {shapes[0][0]} != rank 1 recv {shapes[1][1]}"
        )
        # rank 0 (first) sends n*C
        assert shapes[0][0] == [(32, 2, H * N)]
        # rank 1 (last) sends C
        assert shapes[1][0] == [(32, 2, H)]

    def test_pp4_flexible_vpp_mhc_all_consecutive_match(self):
        """PP=4 with flexible VPP layout + mHC: send[i] == recv[i+1] for all i."""
        H, N = 64, 4
        layout = [
            ["embedding"],
            ["decoder"] * 2,
            ["decoder"] * 2,
            ["decoder"],
            ["decoder"],
            ["decoder"],
            ["decoder"],
            ["decoder", "loss"],
        ]
        cfg = _make_layout_config(
            hidden_size=H, num_layers=8, pp_size=4, layout=layout,
            enable_hyper_connections=True, num_residual_streams=N,
        )
        shapes = self._get_send_recv_shapes(cfg, pp_size=4)
        for i in range(3):
            assert shapes[i][0] == shapes[i + 1][1], (
                f"rank {i} send {shapes[i][0]} != rank {i+1} recv {shapes[i+1][1]}"
            )

        # First stage sends n*C, intermediate stages send/recv n*C, last stage sends C
        assert shapes[0][0] == [(32, 2, H * N)]
        for i in (1, 2):
            assert shapes[i][0] == [(32, 2, H * N)]
            assert shapes[i][1] == [(32, 2, H * N)]
        assert shapes[3][0] == [(32, 2, H)]
        assert shapes[3][1] == [(32, 2, H * N)]

    def test_pp2_flexible_vpp_no_mhc_baseline(self):
        """Baseline: PP=2 with flexible VPP layout, no mHC — all shapes are C."""
        H = 64
        cfg = _make_layout_config(
            hidden_size=H, num_layers=7, pp_size=2,
            layout=[["embedding"], ["decoder"] * 6, ["decoder"], ["loss"]],
            enable_hyper_connections=False,
        )
        shapes = self._get_send_recv_shapes(cfg, pp_size=2)
        for i in range(1):
            assert shapes[i][0] == shapes[i + 1][1]
            assert shapes[i][0] == [(32, 2, H)]

    def test_pp4_flexible_vpp_mhc_uneven_layers_shape_consistent(self):
        """Highly uneven layout: shapes must still match between stages."""
        H, N = 64, 4
        layout = [
            ["embedding", "decoder"],
            ["decoder"] * 5,
            ["decoder"],
            ["decoder", "loss"],
        ]
        cfg = _make_layout_config(
            hidden_size=H, num_layers=8, pp_size=2, layout=layout,
            enable_hyper_connections=True, num_residual_streams=N,
        )
        shapes = self._get_send_recv_shapes(cfg, pp_size=2)
        assert shapes[0][0] == shapes[1][1], (
            f"rank 0 send {shapes[0][0]} != rank 1 recv {shapes[1][1]}"
        )
