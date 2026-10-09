# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The global memory buffer is process-wide scratch memory that needs no parallel_state grid."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _own_tensor_parallel_group(tp_size):
    """Build a tensor-parallel group directly, without parallel_state."""
    if Utils.world_size % tp_size != 0:
        pytest.skip(f"needs a world size divisible by {tp_size}")
    return HyperCommGrid([tp_size, Utils.world_size // tp_size], ["tp", "dp"]).create_pg("tp")


def _small_integers(generator, *shape, low=-2, high=3):
    """Integer-valued float32 data, so that the matmuls below are exact even with TF32."""
    return torch.randint(low, high, shape, generator=generator, device="cuda").float()


class TestGlobalMemoryBuffer:

    def setup_method(self, method):
        Utils.initialize_distributed()
        # No global grid and no buffer: only the groups each test builds exist.
        parallel_state.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("tp_size", [2, 4])
    def test_sequence_parallel_linears_on_their_own_group(self, tp_size):
        """Sequence-parallel linears run on an explicit group without initialize_model_parallel."""
        tp_group = _own_tensor_parallel_group(tp_size)
        tp_rank = tp_group.rank()
        seq_length, batch, hidden, ffn_hidden = 8, 2, 16, 32
        config = ModelParallelConfig(
            tensor_model_parallel_size=tp_size, sequence_parallel=True, perform_initialization=False
        )
        pg_collection = ProcessGroupCollection(tp=tp_group, gtp_remat=None)
        fc1 = ColumnParallelLinear(
            hidden,
            ffn_hidden,
            config=config,
            init_method=torch.nn.init.zeros_,
            bias=False,
            tp_group=tp_group,
            pg_collection=pg_collection,
        )
        fc2 = RowParallelLinear(
            ffn_hidden,
            hidden,
            config=config,
            init_method=torch.nn.init.zeros_,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            tp_group=tp_group,
            pg_collection=pg_collection,
        )

        # The same full tensors on every rank; each rank takes its shards.
        generator = torch.Generator(device="cuda").manual_seed(1234)
        w1 = _small_integers(generator, ffn_hidden, hidden, low=-1, high=2)
        w2 = _small_integers(generator, hidden, ffn_hidden, low=-1, high=2)
        x = _small_integers(generator, seq_length, batch, hidden)
        tokens = slice(tp_rank * seq_length // tp_size, (tp_rank + 1) * seq_length // tp_size)
        features = slice(tp_rank * ffn_hidden // tp_size, (tp_rank + 1) * ffn_hidden // tp_size)
        with torch.no_grad():
            fc1.weight.copy_(w1[features])
            fc2.weight.copy_(w2[:, features])

        x_shard = x[tokens].clone().requires_grad_()
        output, _ = fc2(fc1(x_shard)[0])
        output.sum().backward()

        x_ref, w1_ref, w2_ref = (t.clone().requires_grad_() for t in (x, w1, w2))
        output_ref = x_ref @ w1_ref.t() @ w2_ref.t()
        output_ref.sum().backward()

        exact = dict(rtol=0, atol=0)
        torch.testing.assert_close(output, output_ref[tokens], **exact)
        torch.testing.assert_close(x_shard.grad, x_ref.grad[tokens], **exact)
        torch.testing.assert_close(fc1.weight.grad, w1_ref.grad[features], **exact)
        torch.testing.assert_close(fc2.weight.grad, w2_ref.grad[:, features], **exact)

    def test_gather_into_the_buffer_on_its_own_group(self):
        """The all-gather into the global buffer (MoE allgather dispatcher) runs without the grid."""
        tp_size = 2
        tp_group = _own_tensor_parallel_group(tp_size)
        local = torch.full((2, 3), float(tp_group.rank()), device="cuda")

        gathered = gather_from_sequence_parallel_region(
            local, group=tp_group, use_global_buffer=True
        )

        expected = torch.arange(tp_size, dtype=torch.float32, device="cuda").repeat_interleave(2)
        torch.testing.assert_close(gathered, expected[:, None].expand(-1, 3))

    def test_local_attention_on_its_own_group(self):
        """The local DotProductAttention runs on an explicit group without the global grid."""
        tp_size = 2
        tp_group = _own_tensor_parallel_group(tp_size)
        seq_length, batch, num_heads, head_dim = 8, 2, 4, 8
        config = TransformerConfig(
            num_layers=1,
            hidden_size=num_heads * head_dim,
            num_attention_heads=num_heads,
            tensor_model_parallel_size=tp_size,
            # Sequence parallelism keeps the attention dropout off the model-parallel RNG tracker.
            sequence_parallel=True,
            attention_dropout=0.0,
        )
        attention = DotProductAttention(
            config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=ProcessGroupCollection(tp=tp_group),
        ).cuda()

        generator = torch.Generator(device="cuda").manual_seed(1234)
        shape = (seq_length, batch, num_heads // tp_size, head_dim)
        query, key, value = (
            torch.randn(shape, generator=generator, device="cuda", dtype=torch.float64)
            for _ in range(3)
        )
        output = attention(query, key, value, attention_mask=None)

        scores = torch.einsum("sbnh,tbnh->bnst", query, key) * attention.softmax_scale
        future = torch.ones(seq_length, seq_length, dtype=torch.bool, device="cuda").triu(1)
        probs = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        expected = torch.einsum("bnst,tbnh->sbnh", probs, value).reshape(seq_length, batch, -1)
        torch.testing.assert_close(output, expected)

    def test_buffer_is_created_on_first_use_and_reset_by_destroy(self):
        from megatron.core import utils

        buffer = parallel_state.get_global_memory_buffer()
        assert parallel_state.get_global_memory_buffer() is buffer
        assert utils.get_global_memory_buffer() is buffer

        buffer.get_tensor((4,), torch.float32, "mpu")
        parallel_state.destroy_global_memory_buffer()
        recreated = parallel_state.get_global_memory_buffer()
        assert recreated is not buffer
        assert not recreated.buffer

        parallel_state.destroy_model_parallel()
        assert parallel_state.get_global_memory_buffer() is not recreated

    def test_initialize_model_parallel_keeps_the_existing_buffer(self):
        """A buffer sized before setup (for example, before CUDA-graph capture) stays in place."""
        buffer = parallel_state.get_global_memory_buffer()
        presized = buffer.get_tensor((1024,), torch.float32, "mpu")

        parallel_state.initialize_model_parallel()

        assert parallel_state.get_global_memory_buffer() is buffer
        assert buffer.get_tensor((16,), torch.float32, "mpu").data_ptr() == presized.data_ptr()

    def test_set_global_memory_buffer_keeps_the_existing_buffer(self):
        """Explicitly creating the buffer, as frameworks do before building a model, is harmless."""
        parallel_state._set_global_memory_buffer()
        buffer = parallel_state.get_global_memory_buffer()

        parallel_state._set_global_memory_buffer()

        assert parallel_state.get_global_memory_buffer() is buffer
