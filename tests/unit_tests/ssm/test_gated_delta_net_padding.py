# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.ssm.gated_delta_net import HAVE_FLA
from tests.unit_tests.ssm.gated_delta_net_test_utils import GatedDeltaNetTestBase
from tests.unit_tests.transformer.test_multi_latent_attention import (
    make_test_packed_seq_params,
    make_test_packed_seq_params_with_padding,
)


@pytest.mark.parametrize("use_gdn2", [False, True], ids=["gdn", "gdn2"])
@pytest.mark.parametrize(
    ("tp_size", "sp", "cp_size"),
    [(1, False, 1), (2, False, 1), (2, True, 1), (1, False, 2), (2, False, 2), (2, True, 2)],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.flaky
@pytest.mark.flaky_in_dev
@pytest.mark.internal
class TestGatedDeltaNet(GatedDeltaNetTestBase):
    def test_gpu_forward_thd_padding_correctness(self):
        if self.sp_size > 1:
            pytest.skip("Sequence parallel is not supported for this test case.")

        if self.use_gdn2:
            # See test_gpu_forward_thd_correctness: varlen vs batched kernel paths only
            # match up to bf16 ULP-level differences for GDN2.
            atol, rtol = 1e-2, 1e-2
        else:
            # SBHD and padded THD use distinct FLA bf16 kernel paths, which can
            # differ by one ULP depending on the input and initialized weights.
            atol, rtol = 5e-3, 5e-3
        sequence_length = 32
        micro_batch_size = 4

        # sbhd input shape: [sequence length, batch size, hidden size]
        sub_sequence_length = sequence_length // self.cp_size
        hidden_states_sbhd = torch.rand(
            (sub_sequence_length, micro_batch_size, self.gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        output_sbhd, _ = self.gdn(hidden_states_sbhd, None)

        # thd input shape: [sequence length * batch size, 1, hidden size]
        hidden_states_thd = hidden_states_sbhd.transpose(0, 1).contiguous()
        hidden_states_thd = hidden_states_thd.view(-1, 1, self.gdn.config.hidden_size)
        output_bshd = output_sbhd.transpose(0, 1).contiguous()

        rank = torch.distributed.get_rank()

        # A) padded branch: prefer *_padded when available.
        padded_params = make_test_packed_seq_params_with_padding(
            cu_seqlens=[0, 30, 60, 90, 120], cu_seqlens_padded=[0, 32, 64, 96, 128]
        )
        output_thd_padded, _ = self.gdn(hidden_states_thd, None, packed_seq_params=padded_params)
        output_thd2bshd = output_thd_padded.view(*output_bshd.shape)
        torch.testing.assert_close(
            output_bshd[:, :30, :],
            output_thd2bshd[:, :30, :],
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"THD padded output mismatch ({rank=}): {msg}",
        )

        # B) no-padded branch: use actual cu_seqlens when it matches total_sequence_length.
        no_padding_params = make_test_packed_seq_params(cu_seqlens=[0, 32, 64, 96, 128])
        output_thd_no_padding, _ = self.gdn(
            hidden_states_thd, None, packed_seq_params=no_padding_params
        )
        assert output_thd_no_padding.shape == output_thd_padded.shape

        # C) padded mismatch branch: if *_padded[-1] mismatches total_sequence_length, should raise.
        padded_mismatch_params = make_test_packed_seq_params_with_padding(
            cu_seqlens=[0, 30, 60, 90, 120], cu_seqlens_padded=[0, 32, 64, 96, 126]
        )
        with pytest.raises(ValueError, match="does not match"):
            self.gdn(hidden_states_thd, None, packed_seq_params=padded_mismatch_params)

        # D) actual mismatch branch without *_padded: should raise.
        actual_mismatch_params = make_test_packed_seq_params(cu_seqlens=[0, 32, 64, 96, 129])
        with pytest.raises(ValueError, match="does not match"):
            self.gdn(hidden_states_thd, None, packed_seq_params=actual_mismatch_params)
