# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import patch

import pytest
import torch

from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import common


class TestCommon:

    def test_module_exposes_constants(self):
        """The module exposes GENERATE_NUM and a module-level LOCK."""
        assert common.GENERATE_NUM == 0
        # threading.Lock objects expose acquire/release.
        assert hasattr(common.LOCK, "acquire")
        assert hasattr(common.LOCK, "release")

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="send_do_generate uses cuda.current_device"
    )
    def test_send_do_generate_broadcasts_choice_tensor(self):
        """send_do_generate constructs a [GENERATE_NUM] long tensor on the current device and broadcasts it from rank 0."""
        with patch("megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.common.torch.distributed.broadcast") as fake_bcast:
            common.send_do_generate()
        assert fake_bcast.call_count == 1
        args, kwargs = fake_bcast.call_args
        tensor = args[0]
        src = args[1]
        assert tensor.dtype == torch.long
        assert tensor.tolist() == [common.GENERATE_NUM]
        assert tensor.is_cuda
        assert src == 0
