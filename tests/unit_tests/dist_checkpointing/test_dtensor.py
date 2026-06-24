# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for DTensor support in distributed checkpointing."""

import pytest
import torch

try:
    from torch.distributed.tensor import DeviceMesh, DTensor
    from torch.distributed.tensor.placement_types import Replicate, Shard

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False

from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.strategies.torch import (
    PlaceholderValue,
    convert_state_dict_to_dcp_compatible,
    fill_placeholders,
    inject_placeholders,
    unwrap_dtensors_and_sh_ten,
)
from megatron.core.dist_checkpointing.utils import extract_sharded_base_or_dtensor
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not HAVE_DTENSOR, reason="DTensor not available")


def _get_dtensor_metadata_dp_only():
    """Helper: returns (placements, device_mesh) for DP-only (Replicate) DTensor."""
    from megatron.core.utils import get_dtensor_metadata

    return get_dtensor_metadata(tp=False)


def _make_replicated_sh_ten(key, tensor, replica_id=None):
    """Helper: creates a ShardedTensor with DP-only Replicate DTensor metadata."""
    if replica_id is None:
        replica_id = Utils.rank
    placements, device_mesh = _get_dtensor_metadata_dp_only()
    return ShardedTensor.from_rank_offsets(
        key,
        tensor,
        replica_id=replica_id,
        dtensor_ckpt_device_mesh=device_mesh,
        dtensor_ckpt_placements=placements,
    )


class TestShardedTensorToDTensor:
    """Tests for ShardedTensor.to_dtensor()."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_to_dtensor_replicate(self):
        """ShardedTensor converts to DTensor with Replicate placement preserving values."""
        tensor = torch.full((4, 8), 2.5, device='cuda')
        sh_ten = _make_replicated_sh_ten('test_key', tensor)
        dtensor = sh_ten.to_dtensor()

        assert isinstance(dtensor, DTensor)
        assert torch.equal(dtensor.to_local(), tensor)

    def test_to_dtensor_requires_device_mesh(self):
        """to_dtensor() asserts when dtensor_ckpt_device_mesh is not set."""
        tensor = torch.ones(4, 8, device='cuda')
        sh_ten = ShardedTensor.from_rank_offsets('test_key', tensor, replica_id=0)

        with pytest.raises(AssertionError):
            sh_ten.to_dtensor()

    def test_to_dtensor_preserves_dtype(self):
        """DTensor returned from to_dtensor() has the same dtype as the source."""
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            tensor = torch.ones(3, 5, device='cuda', dtype=dtype)
            sh_ten = _make_replicated_sh_ten(f'key_{dtype}', tensor)
            dtensor = sh_ten.to_dtensor()
            assert dtensor.dtype == dtype

    def test_to_dtensor_tp_shard(self):
        """ShardedTensor converts to DTensor with Shard placement (TP=1 edge-case)."""
        from megatron.core.utils import get_dtensor_metadata

        placements, device_mesh = get_dtensor_metadata(tp=True, tp_axis=0)
        tensor = torch.arange(8, dtype=torch.float, device='cuda').reshape(2, 4)
        sh_ten = ShardedTensor.from_rank_offsets(
            'tp_key',
            tensor,
            (0, 0, 1),  # axis=0, rank_offset=0, total_fragments=1
            replica_id=0,
            dtensor_ckpt_device_mesh=device_mesh,
            dtensor_ckpt_placements=placements,
        )
        dtensor = sh_ten.to_dtensor()
        assert isinstance(dtensor, DTensor)
        assert torch.equal(dtensor.to_local(), tensor)


class TestConvertStateDictToDCP:
    """Tests for convert_state_dict_to_dcp_compatible()."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_converts_sharded_tensors_to_dtensors(self):
        """All ShardedTensors in a flat dict are converted to DTensors."""
        tensor = torch.full((4, 8), 3.0, device='cuda')
        sh_ten = _make_replicated_sh_ten('key_a', tensor)
        state_dict = {'key_a': sh_ten}

        result = convert_state_dict_to_dcp_compatible(state_dict)

        assert isinstance(result['key_a'], DTensor)
        assert torch.equal(result['key_a'].to_local(), tensor)

    def test_converts_multiple_tensors(self):
        """Multiple ShardedTensors are each converted independently."""
        t1 = torch.ones(2, 3, device='cuda') * 1.0
        t2 = torch.ones(5, device='cuda') * 2.0
        state_dict = {
            'k1': _make_replicated_sh_ten('k1', t1),
            'k2': _make_replicated_sh_ten('k2', t2),
        }

        result = convert_state_dict_to_dcp_compatible(state_dict)

        assert isinstance(result['k1'], DTensor)
        assert isinstance(result['k2'], DTensor)
        assert torch.equal(result['k1'].to_local(), t1)
        assert torch.equal(result['k2'].to_local(), t2)

    def test_sharded_object_trivial_shape_unchanged(self):
        """ShardedObjects with trivial global_shape=(1,) are passed through unchanged."""
        sh_obj = ShardedObject('obj_key', b'payload', (1,), (0,), replica_id=0)
        state_dict = {'obj': sh_obj}

        result = convert_state_dict_to_dcp_compatible(state_dict)

        assert result['obj'] is sh_obj

    def test_sharded_object_nontrivial_shape_unchanged(self):
        """ShardedObjects with non-trivial global_shape are returned unchanged (with a log warning)."""
        sh_obj = ShardedObject('obj_key', b'data', (4,), (0,), replica_id=0)
        state_dict = {'obj': sh_obj}

        # Does not raise; logs a warning via the logger (not Python warnings module)
        result = convert_state_dict_to_dcp_compatible(state_dict)

        assert result['obj'] is sh_obj


class TestUnwrapDTensors:
    """Tests for unwrap_dtensors_and_sh_ten()."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_dtensor_to_local_tensor(self):
        """DTensors are unwrapped to plain local tensors."""
        placements, device_mesh = _get_dtensor_metadata_dp_only()
        tensor = torch.full((4, 8), 7.0, device='cuda')
        dtensor = DTensor.from_local(tensor, device_mesh, placements, run_check=False)
        state_dict = {'key': dtensor}

        result = unwrap_dtensors_and_sh_ten(state_dict)

        assert isinstance(result['key'], torch.Tensor)
        assert not isinstance(result['key'], DTensor)
        assert torch.equal(result['key'], tensor)

    def test_multiple_dtensors_unwrapped(self):
        """All DTensors in a flat dict are unwrapped."""
        placements, device_mesh = _get_dtensor_metadata_dp_only()
        t1 = torch.ones(3, device='cuda')
        t2 = torch.full((2, 5), 9.0, device='cuda')
        state_dict = {
            'k1': DTensor.from_local(t1, device_mesh, placements, run_check=False),
            'k2': DTensor.from_local(t2, device_mesh, placements, run_check=False),
        }

        result = unwrap_dtensors_and_sh_ten(state_dict)

        assert torch.equal(result['k1'], t1)
        assert torch.equal(result['k2'], t2)

    def test_sharded_object_data_extracted(self):
        """ShardedObjects are unwrapped to their .data."""
        payload = b'some_bytes'
        sh_obj = ShardedObject('obj_key', payload, (1,), (0,), replica_id=0)
        state_dict = {'obj': sh_obj}

        result = unwrap_dtensors_and_sh_ten(state_dict)

        assert result['obj'] == payload


class TestInjectFillPlaceholders:
    """Tests for inject_placeholders() and fill_placeholders() — no distributed needed."""

    def test_inject_replaces_sharded_tensors_with_placeholders(self):
        """inject_placeholders replaces ShardedTensors with PlaceholderValues."""
        sh_ten_a = ShardedTensor.from_rank_offsets('key_a', torch.ones(2, 3), replica_id=0)
        sh_ten_b = ShardedTensor.from_rank_offsets('key_b', torch.ones(4), replica_id=0)
        state_dict = {'a': sh_ten_a, 'b': sh_ten_b}

        inject_placeholders(state_dict)

        assert isinstance(state_dict['a'], PlaceholderValue)
        assert state_dict['a'].key == 'key_a'
        assert isinstance(state_dict['b'], PlaceholderValue)
        assert state_dict['b'].key == 'key_b'

    def test_inject_returns_flat_dict_of_originals(self):
        """inject_placeholders returns the extracted ShardedTensor objects keyed by their key."""
        sh_ten_a = ShardedTensor.from_rank_offsets('key_a', torch.ones(2, 3), replica_id=0)
        sh_ten_b = ShardedTensor.from_rank_offsets('key_b', torch.ones(4), replica_id=0)
        state_dict = {'a': sh_ten_a, 'b': sh_ten_b}

        extracted = inject_placeholders(state_dict)

        assert set(extracted.keys()) == {'key_a', 'key_b'}
        assert extracted['key_a'] is sh_ten_a
        assert extracted['key_b'] is sh_ten_b

    def test_fill_restores_originals(self):
        """fill_placeholders restores the original objects into the state dict."""
        sh_ten_a = ShardedTensor.from_rank_offsets('key_a', torch.ones(2, 3), replica_id=0)
        state_dict = {'a': sh_ten_a}

        extracted = inject_placeholders(state_dict)
        fill_placeholders(state_dict, extracted)

        assert state_dict['a'] is sh_ten_a

    def test_round_trip_nested_state_dict(self):
        """inject → fill round-trip works with nested dicts."""
        sh_ten_a = ShardedTensor.from_rank_offsets('key_a', torch.ones(2, 3), replica_id=0)
        sh_ten_b = ShardedTensor.from_rank_offsets('key_b', torch.ones(4), replica_id=0)
        state_dict = {'top': sh_ten_a, 'nested': {'inner': sh_ten_b}}

        extracted = inject_placeholders(state_dict)
        fill_placeholders(state_dict, extracted)

        assert state_dict['top'] is sh_ten_a
        assert state_dict['nested']['inner'] is sh_ten_b

    def test_inject_raises_on_duplicate_key(self):
        """inject_placeholders raises RuntimeError when two entries share the same key."""
        sh_ten = ShardedTensor.from_rank_offsets('same_key', torch.ones(2), replica_id=0)
        # Two different state dict entries pointing to ShardedTensors with identical keys
        sh_ten2 = ShardedTensor.from_rank_offsets('same_key', torch.ones(3), replica_id=0)
        state_dict = {'a': sh_ten, 'b': sh_ten2}

        with pytest.raises(RuntimeError, match='Duplicated'):
            inject_placeholders(state_dict)


class TestExtractShardedBaseOrDTensor:
    """Tests for extract_sharded_base_or_dtensor()."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_sharded_tensors_go_to_sharded_part(self):
        """ShardedTensors land in the sharded part, plain values in the common part."""
        sh_ten = ShardedTensor.from_rank_offsets('key_a', torch.ones(2, 3), replica_id=0)
        state_dict = {'sh': sh_ten, 'plain': torch.ones(2)}

        sharded, common = extract_sharded_base_or_dtensor(state_dict)

        assert 'sh' in sharded
        assert 'plain' in common
        assert 'plain' not in sharded
        assert 'sh' not in common

    def test_dtensors_go_to_sharded_part(self):
        """DTensors land in the sharded part alongside ShardedBase objects."""
        placements, device_mesh = _get_dtensor_metadata_dp_only()
        dtensor = DTensor.from_local(
            torch.ones(4, device='cuda'), device_mesh, placements, run_check=False
        )
        state_dict = {'dt': dtensor, 'scalar': 42}

        sharded, common = extract_sharded_base_or_dtensor(state_dict)

        assert 'dt' in sharded
        assert 'scalar' in common
        assert 'dt' not in common
        assert 'scalar' not in sharded

    def test_mixed_state_dict(self):
        """Mixed state dict is split correctly between sharded and common parts."""
        sh_ten = ShardedTensor.from_rank_offsets('k', torch.ones(2), replica_id=0)
        placements, device_mesh = _get_dtensor_metadata_dp_only()
        dtensor = DTensor.from_local(
            torch.ones(3, device='cuda'), device_mesh, placements, run_check=False
        )

        state_dict = {'sh_ten': sh_ten, 'dtensor': dtensor, 'int_val': 99, 'tensor': torch.zeros(5)}

        sharded, common = extract_sharded_base_or_dtensor(state_dict)

        assert set(sharded.keys()) == {'sh_ten', 'dtensor'}
        assert set(common.keys()) == {'int_val', 'tensor'}


class TestGetDTensorMetadata:
    """Tests for get_dtensor_metadata() — verifies placements and mesh structure."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_dp_only_has_replicate_placement(self):
        """Non-TP path returns a single Replicate placement."""
        from megatron.core.utils import get_dtensor_metadata

        placements, _ = get_dtensor_metadata(tp=False)

        assert len(placements) == 1
        assert isinstance(placements[0], Replicate)

    def test_dp_only_mesh_has_dp_dim(self):
        """Non-TP DeviceMesh has 'dp' as its only named dimension."""
        from megatron.core.utils import get_dtensor_metadata

        _, device_mesh = get_dtensor_metadata(tp=False)

        assert isinstance(device_mesh, DeviceMesh)
        assert 'dp' in device_mesh.mesh_dim_names
        assert 'tp' not in device_mesh.mesh_dim_names

    def test_tp_has_shard_and_replicate_placements(self):
        """TP path returns [Shard(tp_axis), Replicate()]."""
        from megatron.core.utils import get_dtensor_metadata

        for tp_axis in (0, 1):
            placements, _ = get_dtensor_metadata(tp=True, tp_axis=tp_axis)

            assert len(placements) == 2
            assert isinstance(placements[0], Shard)
            assert placements[0].dim == tp_axis
            assert isinstance(placements[1], Replicate)

    def test_tp_mesh_has_tp_and_dp_dims(self):
        """TP DeviceMesh has both 'tp' and 'dp' named dimensions."""
        from megatron.core.utils import get_dtensor_metadata

        _, device_mesh = get_dtensor_metadata(tp=True, tp_axis=0)

        assert 'tp' in device_mesh.mesh_dim_names
        assert 'dp' in device_mesh.mesh_dim_names


class TestShardedObjectEmptyFromKey:
    """Tests for ShardedObject.empty_from_key() — no distributed needed."""

    def test_simple_key_creates_trivial_shape(self):
        """A plain key (no slash) creates a ShardedObject with global_shape=(1,)."""
        sh_obj = ShardedObject.empty_from_key('simple_key')

        assert sh_obj.key == 'simple_key'
        assert sh_obj.global_shape == (1,)
        assert sh_obj.global_offset == (0,)
        assert sh_obj.data is None

    def test_valid_unique_key_format_is_parsed(self):
        """A key matching the unique_key format is delegated to empty_from_unique_key."""
        # unique_key format: "<base_key>/shard_<offset>_<shape>"
        sh_obj = ShardedObject.empty_from_key('mykey/shard_3_10')

        assert sh_obj.key == 'mykey'
        assert sh_obj.global_shape == (10,)
        assert sh_obj.global_offset == (3,)

    def test_key_with_slash_invalid_format_falls_back(self):
        """A key with a slash that fails unique_key parsing falls back to (1,) shape."""
        sh_obj = ShardedObject.empty_from_key('mykey/notvalid_0_5')

        assert sh_obj.global_shape == (1,)
        assert sh_obj.global_offset == (0,)

    def test_replica_id_is_forwarded(self):
        """replica_id argument is stored correctly on the resulting ShardedObject."""
        sh_obj = ShardedObject.empty_from_key('some_key', replica_id=7)

        assert sh_obj.replica_id == 7

    def test_returned_object_has_no_data(self):
        """empty_from_key always returns a ShardedObject with data=None."""
        assert ShardedObject.empty_from_key('k').data is None
        assert ShardedObject.empty_from_key('k/shard_0_5').data is None


def _make_dcp_device_mesh():
    """Create a properly-initialized DeviceMesh for the DP group.

    Unlike get_dtensor_metadata() which uses _init_backend=False, this calls
    DeviceMesh.from_group() so that DCP's internal NCCL collectives work correctly.
    """
    from megatron.core import parallel_state

    dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    return DeviceMesh.from_group(dp_group, "cuda")


class TestDTensorSaveLoad:
    """End-to-end save/load tests with use_dtensor_format=True.

    These tests use DeviceMesh.from_group() (not get_dtensor_metadata) so that
    torch.distributed.checkpoint.save/load can use properly-initialized NCCL
    communicators.  get_dtensor_metadata() is tested separately in TestGetDTensorMetadata.
    """

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _make_sh_ten(self, key, tensor, device_mesh, replica_id=None):
        if replica_id is None:
            replica_id = Utils.rank
        return ShardedTensor.from_rank_offsets(
            key,
            tensor,
            replica_id=replica_id,
            dtensor_ckpt_device_mesh=device_mesh,
            dtensor_ckpt_placements=[Replicate()],
        )

    def test_save_load_replicated_tensors_preserves_values(self, tmp_path_dist_ckpt):
        """Round-trip save→load with DTensor format recovers original tensor values."""
        Utils.initialize_model_parallel(1, 1)
        device_mesh = _make_dcp_device_mesh()

        tensor_a = torch.full((4, 8), 2.0, dtype=torch.float32, device='cuda')
        tensor_b = torch.full((3, 5), 3.5, dtype=torch.float32, device='cuda')

        save_sd = {
            'key_a': self._make_sh_ten('key_a', tensor_a, device_mesh),
            'key_b': self._make_sh_ten('key_b', tensor_b, device_mesh),
        }

        with TempNamedDir(tmp_path_dist_ckpt / 'test_replicated') as ckpt_dir:
            save(save_sd, ckpt_dir, async_sharded_save=False, use_dtensor_format=True)

            load_spec = {
                'key_a': self._make_sh_ten('key_a', torch.zeros(4, 8, device='cuda'), device_mesh),
                'key_b': self._make_sh_ten('key_b', torch.zeros(3, 5, device='cuda'), device_mesh),
            }
            loaded = load(load_spec, ckpt_dir, use_dtensor_format=True)

        assert set(loaded.keys()) == {'key_a', 'key_b'}
        assert torch.allclose(loaded['key_a'], tensor_a)
        assert torch.allclose(loaded['key_b'], tensor_b)
        Utils.destroy_model_parallel()

    def test_save_load_bfloat16(self, tmp_path_dist_ckpt):
        """DTensor format round-trip works with bfloat16 tensors."""
        Utils.initialize_model_parallel(1, 1)
        device_mesh = _make_dcp_device_mesh()

        tensor = torch.full((6, 4), 1.5, dtype=torch.bfloat16, device='cuda')
        save_sd = {'bf_key': self._make_sh_ten('bf_key', tensor, device_mesh)}

        with TempNamedDir(tmp_path_dist_ckpt / 'test_bf16') as ckpt_dir:
            save(save_sd, ckpt_dir, async_sharded_save=False, use_dtensor_format=True)

            load_spec = {
                'bf_key': self._make_sh_ten(
                    'bf_key', torch.zeros(6, 4, dtype=torch.bfloat16, device='cuda'), device_mesh
                )
            }
            loaded = load(load_spec, ckpt_dir, use_dtensor_format=True)

        assert loaded['bf_key'].dtype == torch.bfloat16
        assert torch.allclose(loaded['bf_key'].float(), tensor.float())
        Utils.destroy_model_parallel()

    def test_save_load_nested_state_dict(self, tmp_path_dist_ckpt):
        """DTensor format round-trip preserves nested state dict structure."""
        Utils.initialize_model_parallel(1, 1)
        device_mesh = _make_dcp_device_mesh()

        t1 = torch.full((2, 4), 11.0, device='cuda')
        t2 = torch.full((3,), 22.0, device='cuda')

        save_sd = {
            'model': {
                'weight': self._make_sh_ten('model.weight', t1, device_mesh),
                'bias': self._make_sh_ten('model.bias', t2, device_mesh),
            }
        }

        with TempNamedDir(tmp_path_dist_ckpt / 'test_nested') as ckpt_dir:
            save(save_sd, ckpt_dir, async_sharded_save=False, use_dtensor_format=True)

            load_spec = {
                'model': {
                    'weight': self._make_sh_ten(
                        'model.weight', torch.zeros(2, 4, device='cuda'), device_mesh
                    ),
                    'bias': self._make_sh_ten(
                        'model.bias', torch.zeros(3, device='cuda'), device_mesh
                    ),
                }
            }
            loaded = load(load_spec, ckpt_dir, use_dtensor_format=True)

        assert torch.allclose(loaded['model']['weight'], t1)
        assert torch.allclose(loaded['model']['bias'], t2)
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="TP=2 requires at least 2 GPUs")
    def test_save_load_tp2_sharded_tensors(self, tmp_path_dist_ckpt):
        """DTensor format round-trip with TP=2: each rank holds its TP shard."""
        Utils.initialize_model_parallel(2, 1)

        from megatron.core import parallel_state

        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_size = 2
        # dp_rank disambiguates replicas across DP groups (replica_id=0 is the main replica)
        dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)

        dp_size = torch.distributed.get_world_size() // tp_size

        # Build the 2D (tp_size × dp_size) rank tensor by gathering each rank's
        # (global_rank, tp_rank, dp_rank) tuple, then filling mesh[tp_r][dp_r].
        my_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        rank_info = torch.tensor([my_rank, tp_rank, dp_rank], dtype=torch.long, device='cuda')
        all_rank_info = [torch.zeros(3, dtype=torch.long, device='cuda') for _ in range(world_size)]
        torch.distributed.all_gather(all_rank_info, rank_info)
        mesh_tensor = torch.zeros(tp_size, dp_size, dtype=torch.long)
        for info in all_rank_info:
            g_rank, t_rank, d_rank = info.tolist()
            mesh_tensor[int(t_rank)][int(d_rank)] = g_rank

        device_mesh = DeviceMesh("cuda", mesh_tensor, mesh_dim_names=("tp", "dp"))
        placements = [Shard(0), Replicate()]

        # Global tensor rows [0,8) on tp_rank=0, rows [8,16) on tp_rank=1
        local_tensor = torch.arange(
            tp_rank * 8, (tp_rank + 1) * 8, dtype=torch.float32, device='cuda'
        ).reshape(8, 1)

        def make_tp_sh_ten(key, tensor):
            return ShardedTensor.from_rank_offsets(
                key,
                tensor,
                (0, tp_rank, tp_size),
                replica_id=dp_rank,
                dtensor_ckpt_device_mesh=device_mesh,
                dtensor_ckpt_placements=placements,
            )

        save_sd = {'weight': make_tp_sh_ten('weight', local_tensor)}

        with TempNamedDir(tmp_path_dist_ckpt / 'test_tp2') as ckpt_dir:
            save(save_sd, ckpt_dir, async_sharded_save=False, use_dtensor_format=True)

            load_spec = {'weight': make_tp_sh_ten('weight', torch.zeros(8, 1, device='cuda'))}
            loaded = load(load_spec, ckpt_dir, use_dtensor_format=True)

        assert torch.allclose(loaded['weight'], local_tensor)
        Utils.destroy_model_parallel()
