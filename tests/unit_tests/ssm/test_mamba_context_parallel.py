# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import math

import pytest
import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.ssm.mamba_context_parallel import MambaContextParallel
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
class TestMambaContextParallel:

    @pytest.mark.parametrize(
        "ngroups_local_tp, cp_size, D_has_hdim",
        [
            (16, 4, False),  # ngroups_local_tp > cp_size
            (8, 8, False),  # ngroups_local_tp == cp_size
            (4, 8, False),  # ngroups_local_tp < cp_size
            (1, 4, True),  # ngroups_local_tp < cp_size
        ],
    )
    def test_forward(self, ngroups_local_tp, cp_size, D_has_hdim):
        Utils.initialize_model_parallel(context_parallel_size=cp_size)

        dtype = torch.bfloat16
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        headdim = 64
        d_inner_local_tp = cp_size * headdim
        nheads_local_tp = d_inner_local_tp // headdim
        d_state = 128

        conv_dim = d_inner_local_tp + 2 * ngroups_local_tp * d_state
        conv_bias = True
        d_conv = 4
        # weight shape: [conv_dim, 1, d_conv]
        # bias shape: [conv_dim]
        conv1d_cp1 = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            device=device,
            dtype=dtype,
        )

        dt_bias_cp1 = torch.rand(nheads_local_tp, device=device, dtype=dtype)
        A_log_cp1 = torch.rand(nheads_local_tp, device=device, dtype=dtype)
        D_cp1 = torch.rand(
            d_inner_local_tp if D_has_hdim else nheads_local_tp, device=device, dtype=dtype
        )

        cp = MambaContextParallel(
            cp_group=parallel_state.get_context_parallel_group(),
            d_inner_local_tp=d_inner_local_tp,
            nheads_local_tp=nheads_local_tp,
            ngroups_local_tp=ngroups_local_tp,
            d_state=d_state,
            conv1d_cp1=conv1d_cp1,
            dt_bias_cp1=dt_bias_cp1,
            A_log_cp1=A_log_cp1,
            D_cp1=D_cp1,
            D_has_hdim=D_has_hdim,
        )

        sequence_length = cp_size * 2
        batch_size = 1

        # pre_conv_ssm
        sequence_length_cp = sequence_length // cp_size
        in_hidden = 2 * d_inner_local_tp + 2 * ngroups_local_tp * d_state + nheads_local_tp
        in_shape = [sequence_length_cp, batch_size, in_hidden]
        in_tensor = torch.rand(in_shape, device=device, dtype=dtype)
        pre_conv_ssm_tensor = cp.pre_conv_ssm(in_tensor)
        if ngroups_local_tp < cp_size:
            repeat_groups = cp_size // ngroups_local_tp
        else:
            repeat_groups = 1
        repeated_groups_size = ngroups_local_tp * d_state * repeat_groups
        expected_hidden = (
            2 * d_inner_local_tp + 2 * repeated_groups_size + nheads_local_tp
        ) // cp_size
        assert list(pre_conv_ssm_tensor.shape) == [sequence_length, batch_size, expected_hidden]

        d_inner_local_tpcp = d_inner_local_tp // cp_size

        # post_conv_ssm
        y_shape = [sequence_length, batch_size, d_inner_local_tpcp]
        y_tensor = torch.rand(y_shape, device=device, dtype=dtype)
        y_tensor = cp.post_conv_ssm(y_tensor)
        assert list(y_tensor.shape) == [sequence_length_cp, batch_size, d_inner_local_tp]

        # conv1d
        conv_dim_cp = (d_inner_local_tp + 2 * repeated_groups_size) // cp_size
        conv_input_shape = [batch_size, conv_dim_cp, sequence_length]
        conv_input = torch.rand(conv_input_shape, device=device, dtype=dtype)
        conv_output = cp.conv1d(conv_input)
        assert list(conv_output.shape) == [batch_size, conv_dim_cp, sequence_length + d_conv - 1]

        # conv1d_channels
        assert cp.conv1d_channels() == conv_dim_cp

        # get_conv1d_weight
        assert list(cp.get_conv1d_weight().shape) == [conv_dim_cp, 1, d_conv]

        # get_conv1d_bias
        assert list(cp.get_conv1d_bias().shape) == [conv_dim_cp]

        nheads_local_tpcp = nheads_local_tp // cp_size

        # get_dt_bias
        assert list(cp.get_dt_bias().shape) == [nheads_local_tpcp]

        # get_A_log
        assert list(cp.get_A_log().shape) == [nheads_local_tpcp]

        # get_D
        assert list(cp.get_D().shape) == [d_inner_local_tpcp if D_has_hdim else nheads_local_tpcp]

        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "nheads_tp, ngroups_tp, cp_size, expected_error_message",
        [
            (3, 2, 2, "nheads must be evenly divisible by tp_size \\* cp_size"),
            (12, 3, 4, "cp_size must be evenly divisible by ngroups/tp_size"),
            (12, 3, 2, "ngroups must be evenly divisible by tp_size \\* cp_size"),
        ],
    )
    def test_error_check(self, nheads_tp, ngroups_tp, cp_size, expected_error_message):
        Utils.initialize_model_parallel(context_parallel_size=cp_size)
        with pytest.raises(AssertionError, match=expected_error_message):
            cp = MambaContextParallel(
                cp_group=parallel_state.get_context_parallel_group(),
                d_inner_local_tp=nheads_tp,
                nheads_local_tp=nheads_tp,
                ngroups_local_tp=ngroups_tp,
                d_state=None,
                conv1d_cp1=None,
                dt_bias_cp1=None,
                A_log_cp1=None,
                D_cp1=None,
                D_has_hdim=False,
            )
        Utils.destroy_model_parallel()
