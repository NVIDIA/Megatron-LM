# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Process-isolated CUDA graph coverage for the balanced DSv4 CP indexer."""

import pytest
import torch
import torch.distributed as dist

import megatron.core.parallel_state as parallel_state
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
    _SEED,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention_cp import (
    TestDSv4HybridAttentionTHDCP as _DSv4HybridAttentionTHDCPGraphRunners,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention_cp import (
    _clear_cuda_test_state,
    _dsv4_cp_fused_kernels_available,
)

pytestmark = pytest.mark.launch_on_gb200


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSv4HybridAttentionTHDCPCudaGraph:
    """Run graph captures in a dedicated worker process to contain backend state."""

    @pytest.fixture(scope="class", autouse=True, params=(2, 4), ids=lambda cp: f"cp{cp}")
    def setup_method(self, request):
        """Initialize and fully release the test-owned CP communicator."""
        cp_size = request.param
        if Utils.world_size < cp_size:
            pytest.skip(f"THD CP path test requires at least {cp_size} distributed ranks")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        cls = request.cls
        cls.fused_kernels_available = _dsv4_cp_fused_kernels_available()
        cls.cp_size = cp_size
        cls.cp_rank = parallel_state.get_context_parallel_rank()
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()
        cp_group = cls.pg.cp

        yield

        _clear_cuda_test_state()
        Utils.destroy_model_parallel()
        # CUDA graph capture registers buffers with this NCCL communicator. MCore
        # teardown drops its Python references but does not destroy the group, so
        # release it explicitly even though this file runs in an isolated process.
        if cp_group is dist.group.WORLD:
            raise RuntimeError("DSv4 CP graph tests unexpectedly reused the default process group")
        if dist.distributed_c10d._world.pg_map.get(cp_group) is not None:
            dist.destroy_process_group(cp_group)
        cls.pg = None
        _clear_cuda_test_state()

    @pytest.fixture(autouse=True)
    def clear_cuda_test_case(self):
        """Clear CUDA allocator state around each graph runner."""
        _clear_cuda_test_state()
        yield
        _clear_cuda_test_state()

    def test_balanced_dynamic_pack_graph_replays_30_iterations(self):
        """Exercise raw graph route refresh for 30 A/B/C replays."""
        _DSv4HybridAttentionTHDCPGraphRunners.run_balanced_dynamic_pack_graph_replays_30_iterations(
            self
        )

    def test_balanced_dynamic_pack_te_layer_graph_replays_30_iterations(self):
        """Exercise the TE layer route refresh for 30 A/B/C replays."""
        _DSv4HybridAttentionTHDCPGraphRunners.run_balanced_dynamic_pack_te_layer_graph_replays_30_iterations(
            self
        )
