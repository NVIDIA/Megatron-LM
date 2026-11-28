# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""A Mamba state-pool kernel must form its slot offset in int64.

Builds a conv-state pool whose target slot starts past 2^31 elements, passes an int32 slot
index, and checks the kernel touches exactly that slot (bitwise equal to the same kernel on a
one-slot pool). An int32 offset wraps to a wrong address or faults, so the probe runs in a
subprocess. Needs ~4.3 GiB of GPU memory.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_causal_conv1d_update_slot_past_2pow31():
    dim, W = 6144, 4
    S = (2**31) // (dim * (W - 1)) + 8
    device = int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < S * dim * (W - 1) * 2 + (2 << 30):
        pytest.skip("not enough free GPU memory")
    code = f"""
        import torch
        from megatron.core.ssm.ops.common.causal_conv1d_triton import causal_conv1d_update
        # A subprocess starts on GPU 0 unless it explicitly selects its rank's device.
        torch.cuda.set_device({device})
        S, dim, W = {S}, {dim}, {W}
        g = torch.Generator(device="cuda").manual_seed(2)
        big = torch.zeros(S, dim, W - 1, device="cuda", dtype=torch.bfloat16)
        slot = S - 3
        assert slot * big.stride(0) >= 2**31
        init = torch.randn(1, dim, W - 1, device="cuda", generator=g).bfloat16()
        big[slot] = init[0]
        small = init.clone()
        x = torch.randn(1, dim, device="cuda", generator=g).bfloat16()
        w = torch.randn(dim, W, device="cuda", generator=g).bfloat16()
        b = torch.randn(dim, device="cuda", generator=g).bfloat16()
        idx_big = torch.tensor([slot], dtype=torch.int32, device="cuda")
        idx_small = torch.zeros(1, dtype=torch.int32, device="cuda")
        o_big = causal_conv1d_update(x, big, w, b, True, idx_big)
        o_small = causal_conv1d_update(x, small, w, b, True, idx_small)
        torch.cuda.synchronize()
        assert torch.equal(o_big, o_small)
        assert torch.equal(big[slot], small[0])
        assert int(big[: S // 2].count_nonzero()) == 0
        print("PROBE_OK")
        """
    r = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=900
    )
    assert r.returncode == 0 and "PROBE_OK" in r.stdout, (r.stdout[-600:], r.stderr[-1500:])
