import pytest

import transformer_engine as te

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from tests.unit_tests.test_utilities import Utils


class TestTENorm:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=1,
            add_bias_linear=False,
            hidden_size=12,
            num_attention_heads=4,
        )
        yield
        Utils.destroy_model_parallel()

    def test_layer_norm_constructor(self):
        self.transformer_config.normalization = 'LayerNorm'
        instance = TENorm(self.transformer_config, 8)
        assert isinstance(instance, te.pytorch.LayerNorm)

    def test_rms_norm_constructor(self):
        self.transformer_config.normalization = 'RMSNorm'
        instance = TENorm(self.transformer_config, 8)
        assert isinstance(instance, te.pytorch.RMSNorm)

    def test_unknown_constructor(self):
        self.transformer_config.normalization = 'UnknownNorm'
        with pytest.raises(Exception) as e_info:
            TENorm(self.transformer_config, 8)

