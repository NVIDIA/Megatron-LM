import importlib
from types import SimpleNamespace

import torch


def test_default_compressors_exist_for_all_flag_types() -> None:
    import megatron.core.tensor_tracer as tt

    importlib.reload(tt)
    tt._set_compressor()

    for flag_type in tt.FlagType:
        compressor = tt.get_compressor(flag_type)
        assert isinstance(compressor, tt.AbstractCompressor)


def test_ttflags_set_by_configs_sets_flags_and_compressors() -> None:
    import megatron.core.tensor_tracer as tt

    importlib.reload(tt)
    tt._set_compressor()

    args = SimpleNamespace(num_layers=2)
    flags = tt.TTFlags(args)
    flags.set_by_configs(
        {"QKV_mat_mul": "true", "MLP1_mat_mul": "false"},
        {"QKV_mat_mul": {"compressor_type": "NoOpCompressor", "compressor_configs": {}}},
    )

    assert flags.get_flag(tt.FlagType.QKV_mat_mul, 1) is True
    assert flags.get_flag(tt.FlagType.QKV_mat_mul, 2) is True
    assert flags.get_flag(tt.FlagType.MLP1_mat_mul, 1) is False

    assert isinstance(tt.get_compressor(tt.FlagType.QKV_mat_mul), tt.NoOpCompressor)
    empty_compressor = tt.get_compressor(tt.FlagType.MLP1_mat_mul)
    assert isinstance(empty_compressor, tt.EmptyCompressor)

    sample = torch.zeros(2, 3, 4)
    empty_sample = empty_compressor.compress_one_rank(1, tt.FlagType.MLP1_mat_mul, sample)
    assert empty_sample.shape == (2, 3, 0)
    assert empty_sample.device == sample.device


def test_tile_compressor_compress_shapes() -> None:
    import megatron.core.tensor_tracer as tt

    compressor = tt.TileCompressor({"tiles": 4, "tiles_one_rank": 4})
    data = torch.ones(2, 3, 10)
    valid, shape, payload = compressor.compress(1, tt.FlagType.MLP1_mat_mul, data)

    assert valid is True
    assert shape == [2, 3, 4]
    assert payload.numel() == 2 * 3 * 4


def test_tensor_tracers_skips_invalid_compressor_result() -> None:
    import megatron.core.tensor_tracer as tt

    importlib.reload(tt)

    class BadCompressor(tt.AbstractCompressor):
        def compress_one_rank(self, layer_number, flag_type, data):
            return data

        def compress(self, layer_number, flag_type, data):
            return False, [], torch.tensor([])

    tt._GLOBAL_COMPRESSOR = {flag_type: BadCompressor() for flag_type in tt.FlagType}
    called = []
    tt.set_report(lambda name, args, tensor: called.append(name))

    tracer = tt.TensorTracers()
    tracer.report((1, tt.FlagType.QKV_mat_mul), torch.zeros(1, 1, 1))
    assert called == []
