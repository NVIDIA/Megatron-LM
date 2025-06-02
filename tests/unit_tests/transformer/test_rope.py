# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.device_utils import get_current_device_type, get_xla_model
from megatron.core.tensor_parallel.random import model_parallel_device_manual_seed
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
from tests.unit_tests.test_utilities import Utils

xm = get_xla_model()

class TestMultimodalRotaryEmbedding:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_device_manual_seed(123)
        self.kv_channels = 128
        self.rotary_percent = 1.0
        self.rope_device_init = MultimodalRotaryEmbedding(self.kv_channels, self.rotary_percent)

    def teardown_method(self, method):
        del self.rope_device_init
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available() and not xm, reason="Device not available")
    def test_constructor(self):
        assert isinstance(self.rope_device_init, MultimodalRotaryEmbedding)
        assert self.rope_device_init.inv_freq.device.type == get_current_device_type()

    @pytest.mark.skipif(not torch.cuda.is_available() and not xm, reason="Device not available")
    def test_device_forward(self):
        output = self.rope_device_init(torch.Tensor(3, 1, 64), mrope_section=[16, 24, 24])
        assert output.shape[0] == 64
        assert output.shape[1] == 1
        assert output.shape[2] == 1
        assert output.shape[3] == self.kv_channels
        assert output.dtype == torch.float32
        assert output.device.type == get_current_device_type()


class TestRotaryEmbedding:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_device_manual_seed(123)
        self.kv_channels = 8
        self.rotary_percent = 1.0
        self.rope_cpu_init = RotaryEmbedding(
            self.kv_channels, self.rotary_percent, use_cpu_initialization=True
        )
        self.rope_gpu_init = RotaryEmbedding(
            self.kv_channels, self.rotary_percent, use_cpu_initialization=False
        )

    def teardown_method(self, method):
        del self.rope_gpu_init
        del self.rope_cpu_init
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not xm and not torch.cuda.is_available(), reason="Device not available")
    def test_constructor(self):
        assert isinstance(self.rope_cpu_init, RotaryEmbedding)
        assert self.rope_cpu_init.inv_freq.device.type == 'cpu'
        assert isinstance(self.rope_gpu_init, RotaryEmbedding)
        assert self.rope_gpu_init.inv_freq.device.type == get_current_device_type()

    @pytest.mark.skipif(not xm and not torch.cuda.is_available(), reason="Device not available")
    def test_gpu_forward(self):
        output = self.rope_gpu_init(64)
        assert output.shape[0] == 64
        assert output.shape[1] == 1
        assert output.shape[2] == 1
        assert output.shape[3] == self.kv_channels
        assert output.dtype == torch.float32
        assert output.device.type == get_current_device_type()

    @pytest.mark.skipif(not xm and not torch.cuda.is_available(), reason="Device not available")
    def test_cpu_forward(self):
        output = self.rope_cpu_init(64)
        assert output.shape[0] == 64
        assert output.shape[1] == 1
        assert output.shape[2] == 1
        assert output.shape[3] == self.kv_channels
        assert output.dtype == torch.float32
        assert output.device.type == get_current_device_type()
