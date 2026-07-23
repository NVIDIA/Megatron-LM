# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Unit tests for mtp_split pipeline layout support.
# These tests are CPU-only (no GPU or distributed setup required).

import pytest

from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout


class TestMtpSplitLayout:
    """Tests for mtp_split: MTP layers distributed one-per-rank across PP ranks."""

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_layout(layout_str: str, pp_size: int) -> PipelineParallelLayerLayout:
        return PipelineParallelLayerLayout(layout_str, pp_size)

    @staticmethod
    def _validate(layout_str: str, pp_size: int, num_layers: int, mtp_num_layers: int):
        layout = PipelineParallelLayerLayout(layout_str, pp_size)
        return layout.validate_layer_layout(num_layers=num_layers, mtp_num_layers=mtp_num_layers)

    # ------------------------------------------------------------------ #
    # Valid mtp_split layouts                                              #
    # ------------------------------------------------------------------ #

    def test_mtp_split_3layers_8pp(self):
        """E|(t|)*11m|m|m|L  PP=8, VPP=2, mtp_num_layers=3  (the recipe-4 layout).

        Stage count: 16 (= 8 PP × 2 VPP)
        VPP0: PP0=[E], PP1-PP7=[t]  → 7 decoders
        VPP1: PP0-PP3=[t], PP4=[m], PP5=[m], PP6=[m], PP7=[L]  → 4 decoders + 3 mtp
        Total decoders = 7 + 4 = 11
        """
        mtp_standalone = self._validate(
            layout_str="E|(t|)*11m|m|m|L", pp_size=8, num_layers=11, mtp_num_layers=3
        )
        assert mtp_standalone is True, "mtp_split should set mtp_standalone=True"

    def test_mtp_split_2layers_8pp(self):
        """2 MTP layers split across 2 PP ranks.

        Layout: E|(t|)*12m|m|L  PP=8, VPP=2
        VPP0: PP0=[E], PP1-PP7=[t]  → 7 decoders
        VPP1: PP0-PP4=[t], PP5=[m], PP6=[m], PP7=[L]  → 5 decoders + 2 mtp
        Total decoders = 7 + 5 = 12
        """
        mtp_standalone = self._validate(
            layout_str="E|(t|)*12m|m|L", pp_size=8, num_layers=12, mtp_num_layers=2
        )
        assert mtp_standalone is True

    def test_mtp_standalone_single_rank_still_works(self):
        """Existing mtp_standalone (1 MTP on 1 PP rank) must still pass.

        Layout: E|(t|)*13m|L  PP=8, VPP=2
        VPP0: PP0=[E], PP1-PP7=[t]  → 7 decoders
        VPP1: PP0-PP5=[t], PP6=[m], PP7=[L]  → 6 decoders + 1 mtp
        Total decoders = 7 + 6 = 13
        """
        mtp_standalone = self._validate(
            layout_str="E|(t|)*13m|L", pp_size=8, num_layers=13, mtp_num_layers=1
        )
        assert mtp_standalone is True

    def test_mtp_last_rank_not_standalone(self):
        """MTP on the last PP rank → mtp_standalone=False.

        Layout: E|(t|)*14mL  PP=8, VPP=2
        VPP0: PP0=[E], PP1-PP7=[t]  → 7 decoders
        VPP1: PP0-PP6=[t], PP7=[m, L]  → 7 decoders + 1 mtp
        Total decoders = 7 + 7 = 14
        """
        mtp_standalone = self._validate(
            layout_str="E|(t|)*14mL", pp_size=8, num_layers=14, mtp_num_layers=1
        )
        assert mtp_standalone is False

    # ------------------------------------------------------------------ #
    # mtp_split attribute checks                                           #
    # ------------------------------------------------------------------ #

    def test_mtp_split_layout_structure(self):
        """Verify PP rank assignments for "E|(t|)*11m|m|m|L" PP=8."""
        layout = self._make_layout("E|(t|)*11m|m|m|L", pp_size=8)
        from megatron.core.transformer.enums import LayerType

        # VPP0
        assert layout.layout[0][0][0] == LayerType.embedding
        for pp in range(1, 8):
            assert LayerType.decoder in layout.layout[pp][0]

        # VPP1: PP4,PP5,PP6 each have 1 MTP; PP7 has Loss
        for pp in [4, 5, 6]:
            assert layout.layout[pp][1].count(LayerType.mtp) == 1
        assert layout.layout[7][1][0] == LayerType.loss

        # PP4,PP5,PP6 have no MTP in VPP0
        for pp in [4, 5, 6]:
            assert LayerType.mtp not in layout.layout[pp][0]

    # ------------------------------------------------------------------ #
    # get_layer_offset: offsets must be 0, 1, 2 for split ranks          #
    # ------------------------------------------------------------------ #

    def test_mtp_layer_offsets_for_split(self):
        """Each split rank must report a distinct layer offset."""
        from megatron.core.transformer.enums import LayerType

        layout = self._make_layout("E|(t|)*11m|m|m|L", pp_size=8)
        # VPP stage 1 is where MTP lives
        offsets = {
            pp: layout.get_layer_offset(layer_type=LayerType.mtp, vp_stage=1, pp_rank=pp)
            for pp in [4, 5, 6]
        }
        assert offsets[4] == 0, f"PP4 offset should be 0, got {offsets[4]}"
        assert offsets[5] == 1, f"PP5 offset should be 1, got {offsets[5]}"
        assert offsets[6] == 2, f"PP6 offset should be 2, got {offsets[6]}"

    def test_mtp_num_layers_to_build_for_split(self):
        """Each split rank must report 1 MTP layer to build."""
        from megatron.core.transformer.enums import LayerType

        layout = self._make_layout("E|(t|)*11m|m|m|L", pp_size=8)
        for pp in [4, 5, 6]:
            count = layout.get_num_layers_to_build(layer_type=LayerType.mtp, vp_stage=1, pp_rank=pp)
            assert count == 1, f"PP{pp} should build 1 MTP layer, got {count}"

        # Non-MTP ranks build 0
        for pp in [0, 1, 2, 3, 7]:
            count = layout.get_num_layers_to_build(layer_type=LayerType.mtp, vp_stage=1, pp_rank=pp)
            assert count == 0, f"PP{pp} should build 0 MTP layers, got {count}"

    # ------------------------------------------------------------------ #
    # Invalid layouts that must raise                                      #
    # ------------------------------------------------------------------ #

    def test_nonuniform_mtp_split_3plus1(self):
        """Non-uniform split: PP5=[mm] (2 MTP), PP6=[m] (1 MTP) → total 3 MTP.

        Layout: E|(t|)*12mm|m|L  PP=8, VPP=2, mtp_num_layers=3
        VPP0: PP0=[E], PP1-PP7=[t]  → 7 decoders
        VPP1: PP0-PP4=[t], PP5=[mm], PP6=[m], PP7=[L]  → 5 decoders + 3 mtp
        Total decoders = 7 + 5 = 12
        Non-uniform distribution is valid: total count matches mtp_num_layers.
        """
        mtp_standalone = self._validate(
            layout_str="E|(t|)*12mm|m|L", pp_size=8, num_layers=12, mtp_num_layers=3
        )
        assert mtp_standalone is True

    def test_nonuniform_mtp_split_3plus1_offsets(self):
        """Non-uniform split: PP5 has offset=0, PP6 has offset=2 (not 3).

        Offsets count MTP layers before each rank's stage. PP5 holds 2 MTP layers,
        so PP6's offset must be 2 (not 3 as in a uniform split).
        """
        from megatron.core.transformer.enums import LayerType

        layout = self._make_layout("E|(t|)*12mm|m|L", pp_size=8)
        assert layout.get_layer_offset(LayerType.mtp, vp_stage=1, pp_rank=5) == 0
        assert layout.get_layer_offset(LayerType.mtp, vp_stage=1, pp_rank=6) == 2
        assert layout.get_num_layers_to_build(LayerType.mtp, vp_stage=1, pp_rank=5) == 2
        assert layout.get_num_layers_to_build(LayerType.mtp, vp_stage=1, pp_rank=6) == 1

    def test_nonuniform_mtp_split_3plus1_larger(self):
        """Non-uniform split: PP5=[mmm] (3 MTP), PP6=[m] (1 MTP) → total 4 MTP.

        Layout: E|(t|)*12mmm|m|L  PP=8, VPP=2, mtp_num_layers=4
        """
        mtp_standalone = self._validate(
            layout_str="E|(t|)*12mmm|m|L", pp_size=8, num_layers=12, mtp_num_layers=4
        )
        assert mtp_standalone is True

    def test_mtp_in_non_last_vpp_raises(self):
        """MTP must not appear in non-last VPP stage.

        Construct a 16-element list (PP=8, VPP=2) where PP4's VPP0 has MTP.
        Flat list ordering: index = vpp_rank * pp_size + pp_rank
          index 4 = 0*8 + 4 → PP4's VPP0  ← put MTP here
        """
        # 16 elements total: VPP0 (indices 0-7) then VPP1 (indices 8-15)
        layout_list = (
            [["embedding"]]  # index 0:  PP0 VPP0
            + [["decoder"]] * 3  # index 1-3: PP1-PP3 VPP0
            + [["mtp"]]  # index 4:  PP4 VPP0 ← MTP in non-last VPP!
            + [["decoder"]] * 3  # index 5-7: PP5-PP7 VPP0
            + [["decoder"]] * 4  # index 8-11: PP0-PP3 VPP1
            + [["decoder"]]  # index 12: PP4 VPP1
            + [["decoder"]] * 2  # index 13-14: PP5-PP6 VPP1
            + [["loss"]]  # index 15: PP7 VPP1
        )
        # Total: 1+3+1+3+4+1+2+1 = 16, decoders=13, mtp=1
        layout = PipelineParallelLayerLayout(layout_list, pipeline_model_parallel_size=8)
        with pytest.raises(AssertionError):
            layout.validate_layer_layout(num_layers=13, mtp_num_layers=1)
