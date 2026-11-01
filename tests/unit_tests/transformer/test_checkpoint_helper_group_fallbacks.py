# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Checkpoint helpers use their owner's groups; an omitted group falls back with a warning.

Called without groups, the sharded-tensor helpers, ``ensure_metadata_has_dp_cp_group``,
``MegatronModule.sharded_state_dict`` and Mamba's gated norm use the global tensor-parallel and
DP x CP groups. A TE layer built without the replica group of its GTP shards leaves the choice of
each shard's writer to the global grid at checkpoint time. On a model with its own grid, replica
ids then span another grid's ranks. Each of these fallbacks emits a
``ProcessGroupFallbackWarning``, and the in-repo owners pass the groups they hold.
"""

import contextlib
import re
import warnings
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.extensions.transformer_engine import HAVE_TE, _gtp_replica_group
from megatron.core.models.bert.bert_lm_head import BertLMHead
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
    ProcessGroupFallbackWarning,
)
from megatron.core.ssm.mamba_mixer import HAVE_MAMBA_SSM, ExtendedRMSNorm
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.multi_latent_attention import (
    FusedMLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_object_for_checkpoint,
    make_sharded_tensors_for_checkpoint,
)
from megatron.core.utils import (
    make_sharded_tensor_for_checkpoint,
    make_tp_sharded_tensor_for_checkpoint,
)
from tests.unit_tests.test_utilities import (
    Utils,
    forbid_global_process_groups,
    new_group_with_same_ranks,
)

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TELinear
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.models.vision.vit_layer_specs import (
        get_vit_layer_with_transformer_engine_spec,
    )
    from megatron.core.models.vision.vit_model import KimiViTModel, QwenVLViTModel, ViTModel

VERSIONS = "deprecated since Megatron Core 0.21 and will be removed in 0.23"
_FALLBACK_MESSAGE = re.compile(r"^(.+) was called without `([^`]+)`")

needs_tp2 = pytest.mark.skipif(
    Utils.world_size < 2 or Utils.world_size % 2 != 0, reason="needs an even number of ranks"
)
needs_te = pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is not installed")


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


@pytest.fixture
def tp2_grid():
    """The global grid with TP=2; yields the global TP group and GTP-inclusive DP x CP group."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    yield (
        parallel_state.get_tensor_model_parallel_group(),
        parallel_state.get_data_parallel_group(with_context_parallel=True),
    )
    Utils.destroy_model_parallel()


@pytest.fixture
def owner_groups(tp2_grid):
    """Communicators with the ranks of the global TP and DP x CP groups, but not those groups."""
    groups = tuple(new_group_with_same_ranks(group) for group in tp2_grid)
    yield groups
    for group in groups:
        torch.distributed.destroy_process_group(group)


@contextlib.contextmanager
def _no_fallback():
    """Turn every process-group fallback warning into an error inside the block."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProcessGroupFallbackWarning)
        yield


def _fallbacks(record):
    """Return (owner, argument) for each process-group fallback warning in ``record``."""
    found = []
    for warning in record:
        if not issubclass(warning.category, ProcessGroupFallbackWarning):
            continue
        message = str(warning.message)
        assert VERSIONS in message, message
        # The warning points at the code that omitted the groups, not at Megatron Core.
        assert warning.filename == __file__
        found.append(_FALLBACK_MESSAGE.match(message).groups())
    return found


def _shard_ids(entries):
    """``(key, global_shape, global_offset, replica_id)`` of every sharded entry, sorted."""
    if isinstance(entries, dict):
        entries = nested_values(entries)
    return sorted(
        (entry.key, tuple(entry.global_shape), tuple(entry.global_offset), entry.replica_id)
        for entry in entries
    )


def _state(tensor):
    return {'weight': tensor, 'bias': tensor[0], '_extra_state': None}


def _config(**overrides):
    config = dict(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        use_cpu_initialization=True,
        tensor_model_parallel_size=2,
    )
    config.update(overrides)
    return TransformerConfig(**config)


@needs_tp2
class TestShardedHelpers:
    def test_given_groups_are_used_without_reading_the_global_grid(self, owner_groups):
        tp_group, dp_cp_group = owner_groups
        tensor = torch.zeros(4, 4, device='cuda')

        with forbid_global_process_groups(), _no_fallback():
            tp_sharded = make_tp_sharded_tensor_for_checkpoint(
                tensor, 'w', 0, tp_group=tp_group, dp_cp_group=dp_cp_group
            )
            replicated = make_sharded_tensor_for_checkpoint(
                tensor, 'b', tp_group=tp_group, dp_cp_group=dp_cp_group
            )
            sharded_state_dict = make_sharded_tensors_for_checkpoint(
                _state(tensor), 'p.', {'weight': 0}, tp_group=tp_group, dp_cp_group=dp_cp_group
            )
            sharded_object = make_sharded_object_for_checkpoint(None, 'o', replica_id=(0, 1, 2))
            metadata = ensure_metadata_has_dp_cp_group(None, dp_cp_group)

        assert tp_sharded.global_offset == (tp_group.rank() * 4, 0)
        assert tp_sharded.replica_id == (0, 0, dp_cp_group.rank())
        assert replicated.replica_id == (0, tp_group.rank(), dp_cp_group.rank())
        assert sharded_state_dict['p._extra_state'].replica_id == (
            0,
            tp_group.rank(),
            dp_cp_group.rank(),
        )
        assert sharded_object.replica_id == (0, 1, 2)
        assert metadata == {'dp_cp_group': dp_cp_group}

    def test_omitted_groups_warn_and_use_the_global_groups(self, tp2_grid):
        tp_group, dp_cp_group = tp2_grid
        tensor = torch.arange(16.0, device='cuda').view(4, 4)
        groups = dict(tp_group=tp_group, dp_cp_group=dp_cp_group)
        expected = _shard_ids(
            [
                make_tp_sharded_tensor_for_checkpoint(tensor, 'w', 0, **groups),
                make_sharded_tensor_for_checkpoint(tensor, 'b', **groups),
                *make_sharded_tensors_for_checkpoint(
                    _state(tensor), 'p.', {'weight': 0}, **groups
                ).values(),
            ]
        )

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            sharded = [
                make_tp_sharded_tensor_for_checkpoint(tensor, 'w', 0),
                make_sharded_tensor_for_checkpoint(tensor, 'b'),
                *make_sharded_tensors_for_checkpoint(_state(tensor), 'p.', {'weight': 0}).values(),
            ]
            # Each helper warns once per process.
            make_tp_sharded_tensor_for_checkpoint(tensor, 'w', 0)

        assert _shard_ids(sharded) == expected
        assert _fallbacks(record) == [
            ("make_tp_sharded_tensor_for_checkpoint", "tp_group/dp_cp_group"),
            ("make_sharded_tensor_for_checkpoint", "tp_group/dp_cp_group"),
            ("make_sharded_tensors_for_checkpoint", "tp_group/dp_cp_group"),
        ]

    def test_omitted_replica_id_warns_and_uses_the_global_ranks(self, tp2_grid):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            sharded_object = make_sharded_object_for_checkpoint(None, 'o')

        assert sharded_object.replica_id == (
            0,
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )
        assert _fallbacks(record) == [("make_sharded_object_for_checkpoint", "replica_id")]

    def test_metadata_without_a_group_warns_and_names_the_caller(self, tp2_grid):
        _, dp_cp_group = tp2_grid

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            metadata = ensure_metadata_has_dp_cp_group(None)
            # A group already in the metadata wins, so nothing falls back.
            assert ensure_metadata_has_dp_cp_group(metadata) is metadata

        assert metadata['dp_cp_group'] is dp_cp_group
        caller = type(self).test_metadata_without_a_group_warns_and_names_the_caller.__qualname__
        assert _fallbacks(record) == [
            (f"ensure_metadata_has_dp_cp_group (from {caller})", "dp_cp_group")
        ]


class _Leaf(MegatronModule):
    """Own state, and a plain submodule without a sharded_state_dict of its own."""

    def __init__(self, config):
        super().__init__(config)
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.linear = torch.nn.Linear(2, 2)


class _Container(MegatronModule):
    """No state of its own; its only submodule builds its own sharded state dict."""

    def __init__(self, config, child):
        super().__init__(config)
        self.child = child


@needs_tp2
class TestMegatronModuleTensorParallelGroup:
    def test_collection_supplies_the_tp_and_dp_cp_groups(self, owner_groups):
        tp_group, dp_cp_group = owner_groups
        module = _Leaf(_config())
        module.pg_collection = ProcessGroupCollection(tp=tp_group, dp_cp=dp_cp_group)

        with forbid_global_process_groups(), _no_fallback():
            sharded_state_dict = module.sharded_state_dict()

        assert sorted(sharded_state_dict) == ['linear.bias', 'linear.weight', 'weight']
        for entry in sharded_state_dict.values():
            assert entry.replica_id == (0, tp_group.rank(), dp_cp_group.rank())

    def test_module_without_state_reads_no_group(self, owner_groups):
        tp_group, dp_cp_group = owner_groups
        child = _Leaf(_config())
        child.tp_group = tp_group
        container = _Container(_config(), child)

        with forbid_global_process_groups(), _no_fallback():
            sharded_state_dict = container.sharded_state_dict(metadata={'dp_cp_group': dp_cp_group})

        assert sorted(sharded_state_dict) == [
            'child.linear.bias',
            'child.linear.weight',
            'child.weight',
        ]

    def test_module_without_groups_warns_and_uses_the_global_tp_group(self, tp2_grid):
        tp_group, dp_cp_group = tp2_grid
        reference = _Leaf(_config())
        reference.tp_group = tp_group
        expected = _shard_ids(reference.sharded_state_dict(metadata={'dp_cp_group': dp_cp_group}))

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            sharded_state_dict = _Leaf(_config()).sharded_state_dict(
                metadata={'dp_cp_group': dp_cp_group}
            )

        assert _shard_ids(sharded_state_dict) == expected
        assert _fallbacks(record) == [("_Leaf.sharded_state_dict", "tp_group")]


@needs_tp2
@pytest.mark.skipif(not HAVE_MAMBA_SSM, reason="mamba_ssm is not installed")
class TestExtendedRMSNormTensorParallelGroup:
    @staticmethod
    def _norm():
        return ExtendedRMSNorm(
            8,
            eps=1e-5,
            group_size=4,
            norm_before_gate=False,
            device=torch.cuda.current_device(),
            dtype=torch.float32,
        )

    def test_owner_group_is_used(self, owner_groups):
        tp_group, dp_cp_group = owner_groups
        norm = self._norm()
        norm.tp_group = tp_group

        with forbid_global_process_groups(), _no_fallback():
            sharded_state_dict = norm.sharded_state_dict(metadata={'dp_cp_group': dp_cp_group})

        assert sharded_state_dict['weight'].global_offset == (tp_group.rank() * 8,)

    def test_omitted_group_warns_and_uses_the_global_tp_group(self, tp2_grid):
        tp_group, dp_cp_group = tp2_grid
        norm = self._norm()

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            sharded_state_dict = norm.sharded_state_dict(metadata={'dp_cp_group': dp_cp_group})

        assert norm.tp_group is tp_group
        assert sharded_state_dict['weight'].global_offset == (tp_group.rank() * 8,)
        assert _fallbacks(record) == [("ExtendedRMSNorm.sharded_state_dict", "tp_group")]


def _mhc_config():
    return _config(
        num_layers=2,
        hidden_size=64,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=5,
        mhc_init_gating_factor=0.01,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )


class _WrappedLayer(MegatronModule):
    """Stands in for a hybrid layer: the mHC wrapper reads only its number and TP group."""

    def __init__(self, config, tp_group):
        super().__init__(config)
        self.layer_number = 1
        self.tp_group = tp_group


@needs_tp2
class TestOwnersPassTheirTensorParallelGroup:
    def test_bert_lm_head(self, owner_groups):
        tp_group, dp_cp_group = owner_groups
        head = BertLMHead(16, _config(), tp_group=tp_group)

        with forbid_global_process_groups(), _no_fallback():
            sharded_state_dict = head.sharded_state_dict(metadata={'dp_cp_group': dp_cp_group})

        assert head.tp_group is tp_group
        assert sharded_state_dict
        for entry in sharded_state_dict.values():
            assert entry.replica_id == (0, tp_group.rank(), dp_cp_group.rank())

    @needs_te
    def test_hyper_connection_transformer_layer(self, owner_groups):
        tp_group, dp_cp_group = owner_groups
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.tp = tp_group
        spec = get_gpt_layer_with_transformer_engine_spec()
        spec.submodules.self_attention_hyper_connection = HyperConnectionModule
        spec.submodules.mlp_hyper_connection = HyperConnectionModule
        layer = HyperConnectionTransformerLayer(
            _mhc_config(), spec.submodules, layer_number=1, pg_collection=pg_collection
        )

        assert layer.tp_group is tp_group
        for hyper_connection in (layer.self_attention_hyper_connection, layer.mlp_hyper_connection):
            assert hyper_connection.tp_group is tp_group
            with forbid_global_process_groups(), _no_fallback():
                sharded_state_dict = hyper_connection.sharded_state_dict(
                    metadata={'dp_cp_group': dp_cp_group}
                )
            for entry in sharded_state_dict.values():
                assert entry.replica_id == (0, tp_group.rank(), dp_cp_group.rank())

    def test_hyper_connection_hybrid_layer(self, owner_groups):
        tp_group, dp_cp_group = owner_groups
        layer = HyperConnectionHybridLayer(
            config=_mhc_config(), layer=_WrappedLayer(_mhc_config(), tp_group)
        )

        with forbid_global_process_groups(), _no_fallback():
            sharded_state_dict = layer.sharded_state_dict(metadata={'dp_cp_group': dp_cp_group})

        assert layer.hyper_connection.tp_group is tp_group
        assert any(key.startswith('hyper_connection.') for key in sharded_state_dict)
        for entry in sharded_state_dict.values():
            assert entry.replica_id == (0, tp_group.rank(), dp_cp_group.rank())

    @needs_te
    @pytest.mark.parametrize(
        "model_name, kwargs",
        [
            (
                "ViTModel",
                dict(
                    patch_dim=14,
                    img_h=28,
                    img_w=28,
                    add_class_token=True,
                    class_token_len=1,
                    pos_emb_type='learned_absolute',
                ),
            ),
            (
                "QwenVLViTModel",
                dict(patch_dim=14, img_h=28, img_w=28, out_hidden_size=32, num_pos_per_side=2),
            ),
            ("KimiViTModel", dict(img_h=28, img_w=28, pos_embed_height=2, pos_embed_width=2)),
        ],
    )
    def test_vision_encoder(self, owner_groups, model_name, kwargs):
        tp_group, dp_cp_group = owner_groups
        model_class = {
            "ViTModel": ViTModel,
            "QwenVLViTModel": QwenVLViTModel,
            "KimiViTModel": KimiViTModel,
        }[model_name]
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.tp = tp_group
        model = model_class(
            _config(hidden_size=64),
            get_vit_layer_with_transformer_engine_spec(),
            pg_collection=pg_collection,
            **kwargs,
        )

        with forbid_global_process_groups(), _no_fallback():
            sharded_state_dict = model.sharded_state_dict(metadata={'dp_cp_group': dp_cp_group})

        assert model.tp_group is tp_group
        # The patch embedding is outside the decoder and replicated across its TP group.
        replica_id = sharded_state_dict['patch_embed.proj.weight'].replica_id
        assert replica_id == (0, tp_group.rank(), dp_cp_group.rank())

    @needs_te
    def test_fused_mla_fills_metadata_from_its_collection(self, owner_groups):
        _, dp_cp_group = owner_groups
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp_gtp_remat = dp_cp_group
        backend = TESpecProvider()
        config = MLATransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=8,
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_head_dim=32,
            qk_pos_emb_head_dim=32,
            v_head_dim=32,
            rope_type="rope",
            apply_rope_fusion=False,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )
        module = FusedMLASelfAttention(
            config=config,
            submodules=MLASelfAttentionSubmodules(
                linear_q_up_proj=backend.column_parallel_linear(),
                linear_kv_up_proj=backend.column_parallel_linear(),
                core_attention=IdentityOp,
                linear_proj=backend.row_parallel_linear(),
                q_layernorm=IdentityOp,
                kv_layernorm=IdentityOp,
                linear_qkv_down_proj=TELinear,
            ),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=pg_collection,
        )

        with forbid_global_process_groups(), _no_fallback():
            sharded_state_dict = module.sharded_state_dict(prefix='attention.')

        for name in ('linear_q_down_proj', 'linear_kv_down_proj'):
            replica_id = sharded_state_dict[f'attention.{name}.weight'].replica_id
            assert replica_id[-1] == dp_cp_group.rank()


class TestGTPReplicaGroup:
    """The replica group that TE layers stamp on their GTP weight shards.

    No global group is read here: without the group, ``gtp_replica_rank`` falls back to the
    global grid when the checkpoint is saved.
    """

    @staticmethod
    def _gtp_remat_group(size):
        return SimpleNamespace(size=lambda: size)

    def test_layer_without_gtp_needs_no_group(self):
        with _no_fallback():
            for gtp_remat_group in (None, self._gtp_remat_group(1)):
                assert (
                    _gtp_replica_group(None, "dp_cp", gtp_remat_group, owner="TELinearLike") is None
                )

    def test_collection_supplies_the_group(self):
        dense, expert = object(), object()
        pg_collection = ProcessGroupCollection(dp_cp=dense, expt_dp=expert)
        multi_module = MultiModuleProcessGroupCollection(
            module_pgs={"language": pg_collection}, language_model_module_name="language"
        )
        gtp_remat_group = self._gtp_remat_group(2)

        with forbid_global_process_groups(), _no_fallback():
            for collection in (pg_collection, multi_module):
                assert _gtp_replica_group(collection, "dp_cp", gtp_remat_group, owner="X") is dense
                assert (
                    _gtp_replica_group(collection, "expt_dp", gtp_remat_group, owner="X") is expert
                )

    def test_missing_group_warns_and_leaves_the_shards_to_the_global_fallback(self):
        gtp_remat_group = self._gtp_remat_group(2)

        with forbid_global_process_groups(), warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            groups = [
                _gtp_replica_group(
                    ProcessGroupCollection(tp=None), "dp_cp", gtp_remat_group, owner="UnsetLayer"
                ),
                _gtp_replica_group(
                    ProcessGroupCollection(dp_cp=None), "dp_cp", gtp_remat_group, owner="NoneLayer"
                ),
                _gtp_replica_group(None, "expt_dp", gtp_remat_group, owner="ExpertLayer"),
            ]

        assert groups == [None, None, None]
        assert _fallbacks(record) == [
            ("UnsetLayer", "pg_collection.dp_cp"),
            ("NoneLayer", "pg_collection.dp_cp"),
            ("ExpertLayer", "pg_collection.expt_dp"),
        ]
