import unittest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
import math

# Assume the provided class is in mamba_mixer.py
from megatron.core.ssm.mamba_mixer import MambaMixer

class MockContextParallel:
    """
    Mocks the MambaContextParallel helper.
    """
    def __init__(self, d_inner, ngroups, nheads, d_state, device):
        self.d_inner_local_tpcp = d_inner
        self.ngroups_local_tpcp = ngroups
        self.nheads_local_tpcp = nheads
        self.cp_size = 1

        # Random weights for the mock
        self.conv1d_weight = torch.randn(d_inner + 2 * ngroups * d_state, 1, 4, device=device)
        self.conv1d_bias = torch.randn(d_inner + 2 * ngroups * d_state, device=device)
        self.A_log = torch.randn(nheads, device=device)
        self.D = torch.ones(nheads, device=device)
        self.dt_bias = torch.randn(nheads, device=device)

        # Simple conv1d layer for the fallback path if needed
        self.conv1d_layer = nn.Conv1d(
            in_channels=self.conv1d_weight.shape[0],
            out_channels=self.conv1d_weight.shape[0],
            kernel_size=4, groups=self.conv1d_weight.shape[0], padding=3
        ).to(device)

    def get_A_log(self): return self.A_log
    def get_D(self): return self.D
    def get_dt_bias(self): return self.dt_bias
    def get_conv1d_weight(self): return self.conv1d_weight
    def get_conv1d_bias(self): return self.conv1d_bias

    def conv1d(self, x):
        return self.conv1d_layer(x)

    def pre_conv_ssm(self, x): return x
    def post_conv_ssm(self, x): return x


class TestMambaDynamicInference(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cpu':
            self.skipTest("Mamba Triton kernels require CUDA")

        # --- Configuration ---
        self.d_model = 256
        self.d_state = 16
        self.headdim = 64
        self.d_conv = 4
        self.ngroups = 1
        self.d_inner = self.d_model * 2  # expand=2
        self.nheads = self.d_inner // self.headdim

        # Create the Mixer instance directly
        self.mixer = MagicMock(spec=MambaMixer)
        self.mixer.d_state = self.d_state
        self.mixer.d_conv = self.d_conv
        self.mixer.headdim = self.headdim
        self.mixer.chunk_size = 256
        self.mixer.activation = "silu"
        self.mixer.act = nn.SiLU()
        self.mixer.D_has_hdim = False
        self.mixer.rmsnorm = True

        # Mock the Context Parallel wrapper (used by ssm_prefill)
        self.mixer.cp = MockContextParallel(
            d_inner=self.d_inner,
            ngroups=self.ngroups,
            nheads=self.nheads,
            d_state=self.d_state,
            device=self.device
        )

        # --- Setup for ssm_decode ---
        # ssm_decode accesses attributes directly from self, not self.cp
        self.mixer.d_inner_local_tp = self.d_inner
        self.mixer.ngroups_local_tp = self.ngroups
        self.mixer.nheads_local_tp = self.nheads

        # Create real parameters for ssm_decode to access
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.mixer.conv1d = nn.Conv1d(
            in_channels=conv_dim, out_channels=conv_dim,
            kernel_size=self.d_conv, groups=conv_dim, padding=self.d_conv - 1,
            bias=True, device=self.device
        )
        self.mixer.dt_bias = nn.Parameter(torch.randn(self.nheads, device=self.device))
        self.mixer.A_log = nn.Parameter(torch.randn(self.nheads, device=self.device))
        self.mixer.D = nn.Parameter(torch.ones(self.nheads, device=self.device))

        # Bind methods
        self.mixer._ssm_prefill = MambaMixer._ssm_prefill.__get__(self.mixer, MambaMixer)
        self.mixer._ssm_decode = MambaMixer._ssm_decode.__get__(self.mixer, MambaMixer)

    def test_ssm_prefill_padding_isolation(self):
        """
        Tests that ssm_prefill only updates states for the real request
        and outputs zeros for padding tokens.
        """
        num_requests = 48
        real_seq_len = 6
        total_tokens = 63

        # Inputs
        dim_inputs = self.d_inner * 2 + 2 * self.ngroups * self.d_state + self.nheads
        zxBCdt = torch.randn(total_tokens, 1, dim_inputs, device=self.device, dtype=torch.float32)

        # Metadata
        seq_idx = torch.full((total_tokens,), -1, dtype=torch.int32, device=self.device)
        seq_idx[:real_seq_len] = 0
        seq_idx = seq_idx.unsqueeze(0)

        cu_seqlens = torch.full((num_requests + 1,), real_seq_len, dtype=torch.int32, device=self.device)
        cu_seqlens[0] = 0

        batch_indices = torch.full((num_requests,), -1, dtype=torch.long, device=self.device)
        batch_indices[0] = 0

        # States
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        conv_state = torch.zeros(num_requests, conv_dim, self.d_conv, device=self.device)
        ssm_state = torch.zeros(num_requests, self.nheads, self.headdim, self.d_state, device=self.device)

        # Run
        self.mixer.norm = MagicMock(side_effect=lambda x, z: x * z)
        output = self.mixer._ssm_prefill(
            zxBCdt=zxBCdt, conv_state=conv_state, ssm_state=ssm_state,
            seq_idx=seq_idx, cu_seqlens=cu_seqlens,
            batch_indices=batch_indices, return_varlen_states=True
        )

        # Assertions
        real_output = output[0:real_seq_len]
        padding_output = output[real_seq_len:]

        self.assertTrue(torch.allclose(padding_output, torch.zeros_like(padding_output)),
                        "Output for padding tokens should be 0")
        self.assertTrue(conv_state[0].abs().max() > 0, "Real request conv_state should be modified")

        # Verify isolation of padding states
        remaining_conv_states = conv_state[1:num_requests]
        remaining_ssm_states = ssm_state[1:num_requests]

        self.assertTrue(torch.allclose(remaining_conv_states, torch.zeros_like(remaining_conv_states)),
                        "Conv states for padding requests (indices 1 to N-1) should remain 0")
        self.assertTrue(torch.allclose(remaining_ssm_states, torch.zeros_like(remaining_ssm_states)),
                        "SSM states for padding requests (indices 1 to N-1) should remain 0")
        print("Prefill Test Passed!")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
