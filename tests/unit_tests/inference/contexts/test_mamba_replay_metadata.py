# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for `MambaReplayMetadata`, the ReplaySSM state held by the dynamic context.

Geometry and validation are host-side logic. The buffer tests exercise the slot-level
operations the context, controller, and engine delegate to the class; the commit kernel's
full semantics are covered in `tests/unit_tests/ssm/ops/mamba2/test_mamba_ssm_replay.py`, so
here it is only checked that the geometry reaches it correctly.
"""

import pytest
import torch

from megatron.core.inference.config import MambaInferenceStateConfig
from megatron.core.inference.contexts.mamba_replay_metadata import MambaReplayMetadata
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

# 8 heads x 64 dims, d_state 16, 1 group: conv_dim = 8 * 64 + 2 * 1 * 16 = 544.
CONV_SHAPE = (544, 4)
SSM_SHAPE = (8, 64, 16)
MAMBA_HYBRID = [Symbols.MAMBA, Symbols.MLP, Symbols.ATTENTION, Symbols.MLP]
GDN_HYBRID = [Symbols.GDN, Symbols.MLP, Symbols.ATTENTION, Symbols.MLP]


def _state_config(
    layer_type_list=MAMBA_HYBRID, conv_states_shape=CONV_SHAPE, gdp_num_householder=0
) -> MambaInferenceStateConfig:
    return MambaInferenceStateConfig(
        layer_type_list=list(layer_type_list),
        conv_states_shape=conv_states_shape,
        ssm_states_shape=SSM_SHAPE,
        conv_states_dtype=torch.bfloat16,
        ssm_states_dtype=torch.float32,
        gdp_num_householder=gdp_num_householder,
    )


def _create(num_speculative_tokens=2, buffer_len=16, **config_kwargs):
    return MambaReplayMetadata.create(
        enabled=True,
        num_speculative_tokens=num_speculative_tokens,
        buffer_len=buffer_len,
        state_config=_state_config(**config_kwargs),
    )


class TestMambaReplayMetadataGeometry:
    """Construction, validation, and sizing; no buffers are allocated."""

    def test_geometry(self):
        replay = _create(num_speculative_tokens=2, buffer_len=16)
        assert replay.window == 3
        assert replay.flush_threshold == 19  # L = B + W
        assert replay.cache_buf_len == 32  # next power of two >= L
        assert replay.block_spec == 4  # next power of two >= W
        assert replay.ngroups == 1
        assert replay.cache_conv_dim == 8 * 64 + 16  # x|B only; C is never cached

    def test_ring_is_not_padded_when_threshold_is_a_power_of_two(self):
        assert _create(num_speculative_tokens=3, buffer_len=12).cache_buf_len == 16

    def test_ring_bytes_per_request(self):
        replay = _create()
        per_layer = 32 * (528 * torch.bfloat16.itemsize + 8 * torch.float32.itemsize)
        assert replay.ring_bytes_per_request(num_layers=3) == 3 * per_layer

    def test_disabled_returns_none_without_validating(self):
        replay = MambaReplayMetadata.create(
            enabled=False,
            num_speculative_tokens=2,
            buffer_len=16,
            state_config=_state_config(layer_type_list=GDN_HYBRID),
        )
        assert replay is None

    def test_no_speculative_tokens_returns_none_after_validating(self):
        assert _create(num_speculative_tokens=0) is None
        with pytest.raises(ValueError, match="Gated DeltaNet"):
            _create(num_speculative_tokens=0, layer_type_list=GDN_HYBRID)

    @pytest.mark.parametrize(
        "config_kwargs,reason",
        [
            (dict(layer_type_list=GDN_HYBRID), "Gated DeltaNet"),
            (dict(gdp_num_householder=2), "Gated Delta Product"),
            (dict(layer_type_list=[Symbols.ATTENTION, Symbols.MLP]), "no Mamba layers"),
        ],
        ids=["gdn", "gdp", "no_mamba"],
    )
    def test_non_mamba2_models_are_rejected(self, config_kwargs, reason):
        with pytest.raises(ValueError, match=reason):
            _create(**config_kwargs)

    def test_model_without_recurrent_state_is_rejected(self):
        with pytest.raises(ValueError, match="no Mamba layers"):
            MambaReplayMetadata.create(
                enabled=True, num_speculative_tokens=2, buffer_len=16, state_config=None
            )

    def test_buffer_shorter_than_window_is_rejected(self):
        with pytest.raises(ValueError, match="mamba_replay_buffer_len"):
            _create(num_speculative_tokens=4, buffer_len=4)

    @pytest.mark.parametrize(
        "conv_dim,reason",
        [
            (520, "cannot infer ngroups"),  # 520 - 512 = 8 is not a multiple of 2 * d_state
            (608, "not divisible"),  # 608 - 512 = 96 implies 3 groups for 8 heads
        ],
    )
    def test_inconsistent_state_shapes_are_rejected(self, conv_dim, reason):
        with pytest.raises(ValueError, match=reason):
            _create(conv_states_shape=(conv_dim, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ReplaySSM buffers need a GPU")
class TestMambaReplayMetadataBuffers:
    """The slot-level operations the context, controller, and engine delegate to the class."""

    NUM_LAYERS = 2
    MAX_REQUESTS = 4

    def _allocated(self) -> MambaReplayMetadata:
        replay = _create()
        replay.allocate(self.NUM_LAYERS, self.MAX_REQUESTS, torch.cuda.current_device())
        return replay

    def test_allocate_and_deallocate(self):
        replay = self._allocated()
        assert replay.conv_cache.shape == (2, 4, 32, 528)
        assert replay.conv_cache.dtype == torch.bfloat16
        assert replay.dt_cache.shape == (2, 4, 8, 32)
        assert replay.dt_cache.dtype == torch.float32
        for cursor in (replay.write_pos, replay.origin, replay.is_flush):
            assert cursor.shape == (4,) and not bool(cursor.any())
        assert replay.bc_pre.shape == (4, 1, 32, 4)
        assert replay.qsl.tolist() == [0, 3, 6, 9, 12]
        post_conv_cache, dt_cache = replay.layer_caches(1)
        assert post_conv_cache.data_ptr() == replay.conv_cache[1].data_ptr()
        assert dt_cache.data_ptr() == replay.dt_cache[1].data_ptr()

        replay.deallocate()
        for name in MambaReplayMetadata.PERSISTENT_TENSORS + ("bc_pre", "qsl"):
            assert getattr(replay, name) is None

    def test_reset_slots_and_assert_rings_empty(self):
        replay = self._allocated()
        replay.write_pos.fill_(5)
        replay.origin.fill_(3)
        replay.is_flush.fill_(1)
        with pytest.raises(AssertionError, match="non-empty ring"):
            replay.assert_rings_empty([1])

        replay.reset_slots([1, 3])

        assert replay.write_pos.tolist() == [5, 0, 5, 0]
        assert replay.origin.tolist() == [3, 0, 3, 0]
        assert replay.is_flush.tolist() == [1, 0, 1, 0]
        replay.assert_rings_empty([1, 3])

    def test_warmup_commit_leaves_cursors_untouched(self):
        replay = self._allocated()
        replay.write_pos.fill_(2)
        replay.warmup_commit()
        assert replay.write_pos.tolist() == [2, 2, 2, 2]
        assert not bool(replay.is_flush.any())

    def test_commit_advances_decode_rows_and_schedules_flush(self):
        replay = self._allocated()
        replay.write_pos[3] = 12
        device = replay.write_pos.device

        # Rows: slot 3 (2 drafts accepted), slot 0 (none accepted), slot 1 (still in prefill).
        replay.commit(
            accepted_draft_counts=torch.tensor([2, 0, 1], dtype=torch.int64, device=device),
            prefill_status=torch.tensor([0, 0, 1], dtype=torch.int32, device=device),
            state_batch_indices=torch.tensor([3, 0, 1], dtype=torch.int32, device=device),
        )

        # Each decode row commits its accepted drafts plus the bonus token.
        assert replay.write_pos.tolist() == [1, 0, 0, 15]
        # Slot 3 flushes next step: 15 + 2 * W > L = 19, so the next window would not fit.
        assert replay.is_flush.tolist() == [0, 0, 0, 1]
