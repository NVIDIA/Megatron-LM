# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import torch
from torch import nn

from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import (
    get_mcore_tensor_parallel_partition_dim,
    is_mcore_tensor_parallel_duplicated,
    using_tensor_parallel,
)


class DummyMesh:
    def __init__(self, numel: int):
        # mimic DeviceMesh.mesh
        self.mesh = torch.arange(numel)


class DummyDistIndex:
    def __init__(self, tp_dim: str = "tp", numel: int = 1):
        self.tp_dim = tp_dim
        self._is_expert = {}
        self._meshes = {(tp_dim, False): DummyMesh(numel)}

    def get_submesh(self, dim_name: str, is_expert_parallel: bool = False):
        return self._meshes[(dim_name, is_expert_parallel)]


def test_get_mcore_tensor_parallel_partition_dim_column_row_and_none():
    # Column-parallel param -> partition_dim 0
    p_col = torch.nn.Parameter(torch.empty(4, 4))
    p_col._tensor_parallel_mode = "column"
    assert get_mcore_tensor_parallel_partition_dim(p_col) == 0

    # Row-parallel param -> partition_dim 1
    p_row = torch.nn.Parameter(torch.empty(4, 4))
    p_row._tensor_parallel_mode = "row"
    assert get_mcore_tensor_parallel_partition_dim(p_row) == 1

    # Replicated or unknown mode -> None
    p_rep = torch.nn.Parameter(torch.empty(4, 4))
    p_rep._tensor_parallel_mode = "replicated"
    assert get_mcore_tensor_parallel_partition_dim(p_rep) is None

    # No attribute at all -> None
    p_plain = torch.nn.Parameter(torch.empty(4, 4))
    assert get_mcore_tensor_parallel_partition_dim(p_plain) is None


def test_is_mcore_tensor_parallel_duplicated_behaviour():
    # Column / row -> not duplicated (partition_dim not None)
    p_col = torch.nn.Parameter(torch.empty(4, 4))
    p_col._tensor_parallel_mode = "column"
    assert is_mcore_tensor_parallel_duplicated(p_col) is False

    p_row = torch.nn.Parameter(torch.empty(4, 4))
    p_row._tensor_parallel_mode = "row"
    assert is_mcore_tensor_parallel_duplicated(p_row) is False

    # Replicated or no mode -> duplicated == True
    p_rep = torch.nn.Parameter(torch.empty(4, 4))
    p_rep._tensor_parallel_mode = "replicated"
    assert is_mcore_tensor_parallel_duplicated(p_rep) is True

    p_plain = torch.nn.Parameter(torch.empty(4, 4))
    assert is_mcore_tensor_parallel_duplicated(p_plain) is True


def test_using_tensor_parallel_true_when_mesh_size_gt_one():
    # Mesh with >1 element -> using tensor parallel
    dist_index = DummyDistIndex(numel=4)
    assert using_tensor_parallel(dist_index) is True


def test_using_tensor_parallel_false_when_mesh_size_one():
    # Mesh with 1 element -> no tensor parallel
    dist_index = DummyDistIndex(numel=1)
    assert using_tensor_parallel(dist_index) is False


class DummyConfig:
    # Just enough attributes for __init__ to run if needed in future tests.
    gradient_accumulation_fusion = False
    calculate_per_token_loss = False
    init_model_with_meta_device = False
    fp8_recipe = None
    fp8 = False
    gated_linear_unit = False


class DummyDDPConfig:
    # Minimal stub to avoid touching real Megatron FSDP in these tests.
    bucket_size = 1
    grad_reduce_in_fp32 = False
    data_parallel_sharding_strategy = "no_shard"
    num_distributed_optimizer_instances = 1
    outer_dp_sharding_strategy = "no_shard"
    fp8_param_gather = False


def _make_fsdp_for_unit_tests():
    """Construct a FullyShardedDataParallel with minimal stubs.

    We bypass its heavy __init__ by creating an instance via __new__
    and only setting the attributes that _detect_parallelism_type
    and _annotate_tensor_parallelism actually use.
    """
    fsdp = FullyShardedDataParallel.__new__(FullyShardedDataParallel)

    # Copy the registry from the real class.
    fsdp._MODULE_TYPE_REGISTRY = FullyShardedDataParallel._MODULE_TYPE_REGISTRY

    return fsdp


def test_detect_parallelism_telayernormcolumnparallellinear_layernorm_params():
    fsdp = _make_fsdp_for_unit_tests()

    class TELayerNormColumnParallelLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(8, 8))
            self.bias = nn.Parameter(torch.empty(8))
            self.layer_norm_weight = nn.Parameter(torch.empty(8))
            self.layer_norm_bias = nn.Parameter(torch.empty(8))

    module = TELayerNormColumnParallelLinear()

    # layer norm parameters should be replicated
    assert fsdp._detect_parallelism_type("layer_norm_weight", module) == "replicated"
    assert fsdp._detect_parallelism_type("layer_norm_bias", module) == "replicated"

    # non-layer-norm parameters should be column
    assert fsdp._detect_parallelism_type("weight", module) == "column"
    assert fsdp._detect_parallelism_type("bias", module) == "column"


def test_detect_parallelism_registry_column_row_replicated():
    fsdp = _make_fsdp_for_unit_tests()

    # Fabricate simple module classes whose __name__ matches the registry
    class ColumnParallelLinear(nn.Module):
        pass

    class RowParallelLinear(nn.Module):
        pass

    class LayerNorm(nn.Module):
        pass

    assert fsdp._detect_parallelism_type("weight", ColumnParallelLinear()) == "column"
    assert fsdp._detect_parallelism_type("weight", RowParallelLinear()) == "row"
    assert fsdp._detect_parallelism_type("weight", LayerNorm()) == "replicated"


def test_detect_parallelism_tensor_model_parallel_flag_and_partition_dim():
    fsdp = _make_fsdp_for_unit_tests()

    class DummyModule(nn.Module):
        def __init__(self, tensor_model_parallel, partition_dim=None):
            super().__init__()
            self.tensor_model_parallel = tensor_model_parallel
            if partition_dim is not None:
                self.partition_dim = partition_dim

    # tensor_model_parallel = False -> replicated
    m_rep = DummyModule(tensor_model_parallel=False)
    assert fsdp._detect_parallelism_type("weight", m_rep) == "replicated"

    # tensor_model_parallel = True and partition_dim = 0 -> column
    m_col = DummyModule(tensor_model_parallel=True, partition_dim=0)
    assert fsdp._detect_parallelism_type("weight", m_col) == "column"

    # tensor_model_parallel = True and partition_dim = 1 -> row
    m_row = DummyModule(tensor_model_parallel=True, partition_dim=1)
    assert fsdp._detect_parallelism_type("weight", m_row) == "row"


def test_detect_parallelism_norm_fallback():
    fsdp = _make_fsdp_for_unit_tests()

    class MyNormalization(nn.Module):
        pass

    class MyNorm(nn.Module):
        pass

    assert fsdp._detect_parallelism_type("weight", MyNormalization()) == "replicated"
    assert fsdp._detect_parallelism_type("weight", MyNorm()) == "replicated"


def test_detect_parallelism_teliner_parallel_mode_variants():
    fsdp = _make_fsdp_for_unit_tests()

    class TELinear(nn.Module):
        def __init__(self, mode):
            super().__init__()
            self.parallel_mode = mode

    assert fsdp._detect_parallelism_type("weight", TELinear("column")) == "column"
    assert fsdp._detect_parallelism_type("weight", TELinear("row")) == "row"
    assert fsdp._detect_parallelism_type("weight", TELinear("none")) == "replicated"


def test_detect_parallelism_param_level_tp_attributes():
    """Parameters with tensor_model_parallel/partition_dim set directly on them
    (rather than on the owning module) should be detected via the param-level fallback.
    This is the pattern used by MambaMixer's conv1d, A_log, dt_bias, D parameters.
    """
    fsdp = _make_fsdp_for_unit_tests()

    class PlainModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(4, 4))

    module = PlainModule()

    # Without param-level attrs -> None (no inference possible)
    assert fsdp._detect_parallelism_type("weight", module) is None

    # With param-level column (partition_dim=0)
    module.weight.tensor_model_parallel = True
    module.weight.partition_dim = 0
    assert fsdp._detect_parallelism_type("weight", module, module.weight) == "column"

    # With param-level row (partition_dim=1)
    module.weight.partition_dim = 1
    assert fsdp._detect_parallelism_type("weight", module, module.weight) == "row"

    # Row-parallel bias should be replicated
    bias = nn.Parameter(torch.empty(4))
    bias.tensor_model_parallel = True
    bias.partition_dim = 1
    module.bias = bias
    assert fsdp._detect_parallelism_type("bias", module, module.bias) == "replicated"


def test_detect_parallelism_param_level_tp_overrides_norm_fallback():
    """A Norm-like module whose weight has param-level TP attributes should be
    classified by the param-level check, NOT the norm-name fallback.
    This is the pattern used by MambaMixer's ExtendedRMSNorm, whose weight is
    TP-sharded (partition_dim=0) rather than replicated.
    """
    fsdp = _make_fsdp_for_unit_tests()

    class ExtendedRMSNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(8))

    module = ExtendedRMSNorm()

    # Without param-level attrs, norm fallback should return "replicated"
    assert fsdp._detect_parallelism_type("weight", module) == "replicated"

    # With param-level TP attrs, should return "column" instead
    module.weight.tensor_model_parallel = True
    module.weight.partition_dim = 0
    assert fsdp._detect_parallelism_type("weight", module, module.weight) == "column"


def test_detect_parallelism_returns_none_when_cannot_infer():
    fsdp = _make_fsdp_for_unit_tests()

    class PlainModule(nn.Module):
        pass

    assert fsdp._detect_parallelism_type("weight", PlainModule()) is None


def test_annotate_tensor_parallelism_sets_attribute_on_params():
    fsdp = _make_fsdp_for_unit_tests()

    class ColumnParallelLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(4, 4))
            self.bias = nn.Parameter(torch.empty(4))

    class RowParallelLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(4, 4))

    class PlainModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(4, 4))

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.col = ColumnParallelLinear()
            self.row = RowParallelLinear()
            self.plain = PlainModule()

    root = Root()

    # Exercise _annotate_tensor_parallelism
    fsdp._annotate_tensor_parallelism(root)

    # Check that known module types got annotated
    assert root.col.weight._tensor_parallel_mode == "column"
    assert root.col.bias._tensor_parallel_mode == "column"
    assert root.row.weight._tensor_parallel_mode == "row"

    # For unknown module type, _detect_parallelism_type should return None
    # and _annotate_tensor_parallelism must not set the attribute.
    assert not hasattr(root.plain.weight, "_tensor_parallel_mode")


def test_annotate_tensor_parallelism_mamba_mixer_like_module():
    """Simulate a MambaMixer-like module hierarchy where TP attributes are set on
    parameters rather than modules. Verify that _annotate_tensor_parallelism
    correctly classifies all parameters.
    """
    fsdp = _make_fsdp_for_unit_tests()

    class ColumnParallelLinear(nn.Module):
        """Stands in for in_proj (module-level TP, detected via registry)."""

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(8, 4))

    class RowParallelLinear(nn.Module):
        """Stands in for out_proj (module-level TP, detected via registry)."""

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(4, 8))

    class ExtendedRMSNorm(nn.Module):
        """Norm with param-level TP (should NOT fall through to norm fallback)."""

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(8))
            self.weight.tensor_model_parallel = True
            self.weight.partition_dim = 0

    class MambaMixer(nn.Module):
        """Simulates MambaMixer with conv1d and raw TP-sharded parameters."""

        def __init__(self):
            super().__init__()
            self.in_proj = ColumnParallelLinear()
            self.out_proj = RowParallelLinear()
            self.norm = ExtendedRMSNorm()

            # conv1d: standard nn.Conv1d with param-level TP attrs
            self.conv1d = nn.Conv1d(8, 8, 4, groups=8)
            self.conv1d.weight.tensor_model_parallel = True
            self.conv1d.weight.partition_dim = 0
            self.conv1d.bias.tensor_model_parallel = True
            self.conv1d.bias.partition_dim = 0

            # Raw parameters with param-level TP attrs
            self.A_log = nn.Parameter(torch.empty(4))
            self.A_log.tensor_model_parallel = True
            self.A_log.partition_dim = 0

            self.dt_bias = nn.Parameter(torch.empty(4))
            self.dt_bias.tensor_model_parallel = True
            self.dt_bias.partition_dim = 0

            self.D = nn.Parameter(torch.empty(4))
            self.D.tensor_model_parallel = True
            self.D.partition_dim = 0

    mixer = MambaMixer()
    fsdp._annotate_tensor_parallelism(mixer)

    # Module-level detection (via registry)
    assert mixer.in_proj.weight._tensor_parallel_mode == "column"
    assert mixer.out_proj.weight._tensor_parallel_mode == "row"

    # Param-level detection for conv1d
    assert mixer.conv1d.weight._tensor_parallel_mode == "column"
    assert mixer.conv1d.bias._tensor_parallel_mode == "column"

    # Param-level detection for raw parameters on MambaMixer
    assert mixer.A_log._tensor_parallel_mode == "column"
    assert mixer.dt_bias._tensor_parallel_mode == "column"
    assert mixer.D._tensor_parallel_mode == "column"

    # Param-level detection overrides norm fallback
    assert mixer.norm.weight._tensor_parallel_mode == "column"
