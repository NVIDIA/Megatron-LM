import torch

from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb_with_cos_sin
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding


class TestRotaryEmbeddingWithPrecomputedCosSin:

    def setup_method(self):
        self.batch_size = 3
        self.seq_len = 4
        self.d_rot = 6
        self.rotary_embedding = RotaryEmbedding(kv_channels=4, rotary_percent=1.0)

    def test_output_shapes_match(self):

        # Create input tensors
        t = torch.randn(self.seq_len, self.batch_size, 2, self.d_rot * 2, device="cuda")
        rotary_pos_cos, rotary_pos_sin = self.rotary_embedding.get_cos_sin(self.seq_len)

        # Test using Flash Decoding optimized kernel which requires precomputed cos & sin tensors
        expected_shape = torch.Size(
            [self.seq_len, self.batch_size, self.seq_len // 2, self.seq_len * self.batch_size]
        )
        output_flash_rotary = apply_rotary_pos_emb_with_cos_sin(
            t, rotary_pos_cos, rotary_pos_sin, rotary_interleaved=True
        )

        assert (
            output_flash_rotary.shape == expected_shape
        ), f"Outputs do not match: {output_flash_rotary.shape} != {expected_shape}"
