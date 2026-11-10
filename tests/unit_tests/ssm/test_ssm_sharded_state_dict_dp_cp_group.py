# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""SSM mixers and hybrid stacks compute replica ids over their own DP x CP group.

Called without metadata, ``sharded_state_dict`` filled in the global
``get_data_parallel_group(with_context_parallel=True)``. A model on its own grid then claimed
replicas by another grid's DP ranks, so several ranks or none wrote each shard.
"""

import pytest
import torch.nn.functional as F

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import gdp_stack_spec, hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm import gated_delta_product
from megatron.core.ssm.gated_delta_net import HAVE_FLA, GatedDeltaNet
from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.ssm.gdp_test_utils import HAVE_GDP_DEPS, _make_config
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_sharded_state_dict_dp_cp_group import (
    forbid_global_dp_cp_group,
    shard_ids,
)


class TestSSMShardedStateDictDpCpGroup:

    def setup_method(self, method):
        if Utils.world_size < 2:
            pytest.skip("needs at least two data-parallel ranks")
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _config(num_layers):
        # The Mamba mixer places several constraints on the hidden size.
        return TransformerConfig(
            hidden_size=256,
            num_layers=num_layers,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )

    def _assert_own_group_without_global_read(self, module):
        """Same keys, offsets and replica ids as with the module's own group, no global read."""
        own_metadata = {'dp_cp_group': self.pg_collection.dp_cp_gtp_remat}
        expected = shard_ids(module.sharded_state_dict(metadata=own_metadata))

        with forbid_global_dp_cp_group():
            sharded_state_dict = module.sharded_state_dict()

        assert shard_ids(sharded_state_dict) == expected

    def test_mamba_mixer_without_metadata(self):
        config = self._config(num_layers=1)
        mixer = MambaMixer(
            config,
            hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
            config.hidden_size,
            layer_number=1,
            pg_collection=self.pg_collection,
        ).cuda()

        self._assert_own_group_without_global_read(mixer)

    def test_hybrid_stack_without_metadata(self):
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        config = self._config(num_layers=len(layer_pattern))
        stack = HybridStack(
            config,
            hybrid_stack_spec.submodules,
            layer_config_list=validate_segment_layers(layer_pattern, config),
            pp_layer_offset=0,
            pg_collection=self.pg_collection,
        ).cuda()

        self._assert_own_group_without_global_read(stack)

    @pytest.mark.skipif(not HAVE_FLA, reason="GDN needs FLA")
    def test_gated_delta_net_without_metadata(self):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=2,
            num_query_groups=2,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            normalization="RMSNorm",
            layernorm_zero_centered_gamma=True,
            activation_func=F.silu,
            use_cpu_initialization=True,
            bf16=True,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        gdn = GatedDeltaNet(
            config,
            submodules=get_experimental_attention_variant_module_spec(config=config).submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=self.pg_collection,
        ).cuda()

        self._assert_own_group_without_global_read(gdn)

    @pytest.mark.skipif(not HAVE_GDP_DEPS, reason="requires GDP dependencies")
    def test_gdp_mixer_gives_its_norm_the_dp_cp_group(self):
        """The norm's own ``sharded_state_dict`` uses the group that the mixer set on it."""
        config = _make_config([])
        config.gdp_cutedsl_kernel = not gated_delta_product.HAVE_FLA
        mixer = GatedDeltaProductMixer(
            config,
            gdp_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
            config.hidden_size,
            layer_number=1,
            pg_collection=self.pg_collection,
        ).cuda()

        assert mixer.norm.dp_cp_group is self.pg_collection.dp_cp_gtp_remat
        self._assert_own_group_without_global_read(mixer.norm)
