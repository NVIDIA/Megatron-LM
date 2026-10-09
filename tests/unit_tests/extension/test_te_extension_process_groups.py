# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Process groups used by the Transformer Engine fused MLP and dot-product attention wrappers."""

import contextlib

import pytest
import torch
import torch.nn.functional as F

import megatron.core.extensions.transformer_engine as te_ext
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import (
    HAVE_TE,
    TEDotProductAttention,
    TEFusedMLP,
    TEFusedMLPWithGroupedLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

# Name suffixes of the parallel_state accessors that return a group, a rank or a size.
_ACCESSOR_SUFFIXES = ("_group", "_groups", "_gloo", "_rank", "_ranks", "_world_size")

_FUSED_MLP_CLASSES = [
    pytest.param(
        TEFusedMLP,
        marks=pytest.mark.skipif(TEFusedMLP is None, reason="TE operation-based MLP unavailable"),
        id="TEFusedMLP",
    ),
    pytest.param(
        TEFusedMLPWithGroupedLinear,
        marks=pytest.mark.skipif(
            TEFusedMLPWithGroupedLinear is None or not is_te_min_version("2.14.0"),
            reason="TEFusedMLPWithGroupedLinear requires Transformer Engine >= 2.14.0",
        ),
        id="TEFusedMLPWithGroupedLinear",
    ),
]


@contextlib.contextmanager
def _forbid_global_grid():
    """Make the parallel_state group, rank and size accessors raise inside the block.

    The TE extension imports some accessors by name, so they are replaced there as well.
    """

    def forbidden(name):
        def read_global_grid(*args, **kwargs):
            raise AssertionError(f"parallel_state.{name}() reads the global parallel grid")

        return read_global_grid

    with pytest.MonkeyPatch.context() as patch:
        for name, value in list(vars(parallel_state).items()):
            if callable(value) and name.startswith("get_") and name.endswith(_ACCESSOR_SUFFIXES):
                stub = forbidden(name)
                patch.setattr(parallel_state, name, stub)
                if name in vars(te_ext):
                    patch.setattr(te_ext, name, stub)
        yield


def _strided_tp_group(tp_size):
    """Return this rank's TP group of a grid whose TP groups stride over the ranks.

    The first grid dimension varies fastest, so consecutive ranks are in different TP groups,
    unlike in the global grid.
    """
    world_size = torch.distributed.get_world_size()
    grid = HyperCommGrid([world_size // tp_size, tp_size], ["dp", "tp"])
    return grid.create_pg("tp")


class TestTEFusedMLPTensorParallelGroup:

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("fused_mlp_class", _FUSED_MLP_CLASSES)
    @pytest.mark.parametrize(("global_tp_size", "tp_size"), [(2, 2), (1, 4)])
    @pytest.mark.parametrize("sequence_parallel", [False, True])
    def test_matches_unfused_mlp_on_its_tp_group(
        self, fused_mlp_class, global_tp_size, tp_size, sequence_parallel
    ):
        """The fused MLP communicates over its own TP group, not the global one."""
        if Utils.world_size < 4 or Utils.world_size % tp_size:
            pytest.skip(f"needs at least 4 ranks, divisible by {tp_size}")
        Utils.initialize_model_parallel(tensor_model_parallel_size=global_tp_size)
        model_parallel_cuda_manual_seed(123)

        tp_group = _strided_tp_group(tp_size)
        # gtp_remat=None stops the linear layers from looking up GTP groups in the global grid.
        pg_collection = ProcessGroupCollection(tp=tp_group, gtp_remat=None)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            ffn_hidden_size=128,
            num_attention_heads=4,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=sequence_parallel,
            gated_linear_unit=True,
            activation_func=F.silu,
            add_bias_linear=False,
            normalization="RMSNorm",
            init_method_std=0.1,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        submodules = MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
        )

        torch.manual_seed(1234)
        hidden_states = torch.randn(16, 2, config.hidden_size, device="cuda")
        if sequence_parallel:
            hidden_states = hidden_states.chunk(tp_size)[tp_group.rank()]
        output_grad = torch.randn_like(hidden_states)

        results = []
        with _forbid_global_grid():
            mlp = MLP(
                config,
                submodules,
                ffn_hidden_size=config.ffn_hidden_size,
                tp_group=tp_group,
                pg_collection=pg_collection,
            )
            fused_mlp = fused_mlp_class(
                config,
                submodules,
                ffn_hidden_size=config.ffn_hidden_size,
                tp_group=tp_group,
                pg_collection=pg_collection,
            )
            with torch.no_grad():
                for (name, param), (fused_name, fused_param) in zip(
                    mlp.named_parameters(), fused_mlp.named_parameters()
                ):
                    assert name == fused_name
                    fused_param.copy_(param)

            for module in (mlp, fused_mlp):
                inputs = hidden_states.clone().requires_grad_()
                output, bias = module(inputs)
                assert bias is None
                output.backward(output_grad)
                results.append(
                    (output, inputs.grad, {n: p.grad for n, p in module.named_parameters()})
                )

        unfused_results, fused_results = results
        # With sequence parallelism, the two paths sum some gradient terms in a different order.
        torch.testing.assert_close(fused_results, unfused_results, rtol=1e-4, atol=1e-3)


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is not installed")
class TestTEDotProductAttentionContextParallelGroup:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _tp_and_cp_groups():
        world_size = torch.distributed.get_world_size()
        grid = HyperCommGrid([1, 2, world_size // 2], ["tp", "cp", "dp"])
        return grid.create_pg("tp"), grid.create_pg("cp")

    @staticmethod
    def _build(pg_collection):
        config = TransformerConfig(
            num_layers=1, hidden_size=64, num_attention_heads=4, context_parallel_size=2
        )
        return TEDotProductAttention(
            config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=pg_collection,
        )

    @pytest.mark.parametrize("cp_field", ["absent", "none"])
    def test_rejects_context_parallelism_without_cp_group(self, cp_field):
        """A missing CP group must not turn every rank of the job into the CP ranks."""
        if Utils.world_size % 2:
            pytest.skip("needs an even number of ranks")
        tp_group, _ = self._tp_and_cp_groups()
        if cp_field == "absent":
            pg_collection = ProcessGroupCollection(tp=tp_group)
        else:
            pg_collection = ProcessGroupCollection(tp=tp_group, cp=None)

        with pytest.raises(ValueError, match="pg_collection.cp"):
            self._build(pg_collection)

    def test_uses_the_cp_group_of_the_collection(self):
        if Utils.world_size % 2:
            pytest.skip("needs an even number of ranks")
        tp_group, cp_group = self._tp_and_cp_groups()

        attention = self._build(ProcessGroupCollection(tp=tp_group, cp=cp_group))

        assert attention.cp_group is cp_group
        assert attention.cp_global_ranks == torch.distributed.get_process_group_ranks(cp_group)
