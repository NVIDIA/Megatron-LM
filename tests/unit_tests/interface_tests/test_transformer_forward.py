import inspect

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.unit_tests.test_utilities import Utils


class TestTransformerLayerInterface:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=1, hidden_size=4, num_attention_heads=4, use_cpu_initialization=True
        )

        self.config = TransformerConfig(hidden_size=8, num_attention_heads=1, num_layers=1)
        self.submodules = TransformerLayerSubmodules()
        self.layer = TransformerLayer(self.config, self.submodules)

    def test_forward_args(self):
        # Get the signature of the forward method
        forward_signature = inspect.signature(self.layer.forward)

        # Define the expected parameter names
        expected_params = [
            'hidden_states',
            'attention_mask',
            'context',
            'context_mask',
            'rotary_pos_emb',
            'rotary_pos_cos',
            'rotary_pos_sin',
            'inference_params',
            'packed_seq_params',
        ]
        # Check if the parameter names match the expected names
        assert (
            list(forward_signature.parameters.keys()) == expected_params
        ), "TransformerLayer.forward() interface has changed!"
