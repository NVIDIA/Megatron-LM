# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest import mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.ssm.utils import _split_tensor_factory
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.ckpt import native
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
    setup_model_and_optimizer,
)
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("nvrx_checkpointing")


def _state_dict(step, extra=False):
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    dp_rank = parallel_state.get_data_parallel_rank()
    weight = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    model = {
        "weight": ShardedTensor.from_rank_offsets(
            "model.weight", weight + 100 * tp_rank + step, (0, tp_rank, tp_size), replica_id=dp_rank
        )
    }
    if extra:
        model["bias"] = ShardedTensor.from_rank_offsets(
            "model.bias", torch.full((4,), float(step), device="cuda"), replica_id=Utils.rank
        )
    return {
        "model": model,
        "rng_state": ShardedObject(
            "rng_state", list(range(step)), (Utils.world_size,), (Utils.rank,)
        ),
        "iteration": step,
    }


def _load(checkpoint_dir):
    template = _state_dict(0)
    template["model"]["weight"].data = torch.empty(4, 4, device="cuda")
    return load(template, checkpoint_dir)


def _packed(encoding, rows, cols, seed):
    """A bf16 weight packed as MXFP8 or row-scaled NVFP4 from random bytes, with its values.

    No Transformer Engine kernel runs, so GPUs without MXFP8 or NVFP4 support can save it too.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def random_bytes(shape, low=0, high=256):
        return torch.randint(
            low, high, shape, dtype=torch.uint8, device="cuda", generator=generator
        )

    def padded(scales):
        shape = (-(-rows // 128) * 128, -(-scales.shape[1] // 4) * 4)
        result = torch.zeros(shape, dtype=torch.uint8, device="cuda")
        result[:rows, : scales.shape[1]] = scales
        return result

    if encoding == "mxfp8":
        mxfp8_tensor = pytest.importorskip("transformer_engine.pytorch.tensor.mxfp8_tensor")
        codes = random_bytes((rows, cols))
        codes[(codes & 0x7F) == 0x7F] = 0
        scales = random_bytes((rows, cols // 32), 120, 134)
        tensor = mxfp8_tensor.MXFP8Tensor(
            shape=(rows, cols),
            dtype=torch.bfloat16,
            rowwise_data=codes,
            rowwise_scale_inv=padded(scales),
            columnwise_data=None,
            columnwise_scale_inv=None,
            fp8_dtype=mxfp8_tensor.tex.DType.kFloat8E4M3,
            quantizer=None,
            with_gemm_swizzled_scales=False,
        )
        values = codes.view(torch.float8_e4m3fn).float()
        values *= torch.exp2(scales.float() - 127).repeat_interleave(32, 1)
    else:
        nvfp4_tensor = pytest.importorskip("transformer_engine.pytorch.tensor.nvfp4_tensor")
        codes = random_bytes((rows, cols // 2))
        scales = random_bytes((rows, cols // 16), 48, 72)
        amax = torch.rand(rows, device="cuda", generator=generator) * 8 + 1
        tensor = nvfp4_tensor.NVFP4Tensor(
            shape=(rows, cols),
            dtype=torch.bfloat16,
            rowwise_data=codes,
            rowwise_scale_inv=padded(scales),
            columnwise_data=None,
            columnwise_scale_inv=None,
            amax_rowwise=amax,
            amax_columnwise=None,
            fp4_dtype=nvfp4_tensor.tex.DType.kFloat4E2M1,
            quantizer=None,
            with_gemm_swizzled_scales=False,
            row_scaled_nvfp4=True,
        )
        e2m1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
        nibbles = torch.stack((codes & 0xF, codes >> 4), dim=-1).reshape(rows, cols).long()
        block_scales = scales.view(torch.float8_e4m3fn).float() * amax[:, None]
        block_scales *= 1 / torch.tensor(6 * 448.0, device="cuda")
        values = torch.tensor(e2m1, device="cuda")[nibbles] * block_scales.repeat_interleave(16, 1)
    return tensor, values.bfloat16()


class _Block(MegatronModule):
    """A SwiGLU MLP and a Mamba-style input projection, saved through their checkpoint factories."""

    def __init__(self, in_proj, fc1=None, fc2=None):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            ffn_hidden_size=256,
            num_attention_heads=1,
            gated_linear_unit=True,
            add_bias_linear=False,
            bias_activation_fusion=False,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
        )
        super().__init__(config=config)
        self.mlp = MLP(config, get_submodules(get_gpt_layer_local_submodules().mlp)).cuda()
        for linear, weight in ((self.mlp.linear_fc1, fc1), (self.mlp.linear_fc2, fc2)):
            if weight is not None:
                linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.in_proj = torch.nn.Parameter(in_proj, requires_grad=False)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        state = self.mlp.sharded_state_dict(f"{prefix}mlp.", sharded_offsets, metadata)
        in_proj = ShardedTensor.from_rank_offsets(
            f"{prefix}in_proj",
            self.in_proj,
            (
                0,
                parallel_state.get_tensor_model_parallel_rank(),
                parallel_state.get_tensor_model_parallel_world_size(),
            ),
            replica_id=parallel_state.get_data_parallel_rank(),
        )
        state[f"{prefix}in_proj"] = _split_tensor_factory(in_proj, [128, 256], ["z", "x"], 0)
        return state


class TestNativeCheckpointing:
    def setup_method(self, method):
        Utils.initialize_model_parallel(2, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_saves_store_the_values_of_each_save(self, tmp_path_dist_ckpt):
        with TempNamedDir(tmp_path_dist_ckpt / "native_values") as checkpoint_dir:
            for step in (1, 2):
                native.save(_state_dict(step), checkpoint_dir / str(step))()
            for step in (1, 2):
                loaded = _load(checkpoint_dir / str(step))
                expected = _state_dict(step)
                assert torch.equal(loaded["model"]["weight"], expected["model"]["weight"].data)
                assert loaded["rng_state"] == list(range(step))
                assert loaded["iteration"] == step

    def test_a_registered_state_is_saved_without_registering_again(self, tmp_path_dist_ckpt):
        with TempNamedDir(tmp_path_dist_ckpt / "native_registered") as checkpoint_dir:
            native.register(_state_dict(1))
            session_id = native._session_id
            native.save(_state_dict(2), checkpoint_dir)()
            assert native._session_id == session_id
            loaded = _load(checkpoint_dir)
            assert torch.equal(loaded["model"]["weight"], _state_dict(2)["model"]["weight"].data)
            assert loaded["rng_state"] == [0, 1] and loaded["iteration"] == 2

    def test_a_state_with_another_structure_is_registered_anew(self, tmp_path_dist_ckpt):
        with TempNamedDir(tmp_path_dist_ckpt / "native_restructured") as checkpoint_dir:
            native.save(_state_dict(1), checkpoint_dir / "1")()
            session_id = native._session_id
            native.save(_state_dict(2, extra=True), checkpoint_dir / "2")()
            assert native._session_id != session_id
            template = _state_dict(0, extra=True)
            loaded = load(template, checkpoint_dir / "2")
            assert torch.equal(loaded["model"]["bias"], torch.full((4,), 2.0, device="cuda"))

    def test_lists_of_sharded_tensors_and_lists_are_saved(self, tmp_path_dist_ckpt):
        # Like the fp32 params of a mixed-precision optimizer, where SwiGLU builds [w, v] lists.
        def state(value):
            halves = [
                ShardedTensor.from_rank_offsets(
                    "fc1",
                    torch.full((2, 4), value + half, device="cuda"),
                    (0, half, 2),
                    replica_id=Utils.rank,
                )
                for half in range(2)
            ]
            proj = ShardedTensor.from_rank_offsets(
                "proj", torch.full((4,), value, device="cuda"), replica_id=Utils.rank
            )
            return {"optimizer": {0: {"fp32_from_fp16_params": [[proj, halves]]}}}

        with TempNamedDir(tmp_path_dist_ckpt / "native_nested_lists") as checkpoint_dir:
            native.save(state(1.0), checkpoint_dir)()
            loaded = load(state(0.0), checkpoint_dir)["optimizer"][0]["fp32_from_fp16_params"]
            assert torch.equal(loaded[0][0], torch.full((4,), 1.0, device="cuda"))
            assert torch.equal(loaded[0][1][1], torch.full((2, 4), 2.0, device="cuda"))

    def test_quantized_tensors_that_are_not_kept_packed_are_saved_dequantized(
        self, tmp_path_dist_ckpt
    ):
        blockwise = pytest.importorskip("transformer_engine.pytorch.tensor.float8_blockwise_tensor")
        supported, reason = pytest.importorskip(
            "transformer_engine.pytorch.fp8"
        ).check_fp8_block_scaling_support()
        if not supported:
            pytest.skip(reason)
        quantizer = blockwise.Float8BlockQuantizer(
            fp8_dtype=blockwise.tex.DType.kFloat8E4M3, rowwise=True, columnwise=False
        )
        generator = torch.Generator(device="cuda").manual_seed(0)
        values = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
        weight = quantizer(values)

        def state(data):
            return {"w": ShardedTensor.from_rank_offsets("w", data, replica_id=Utils.rank)}

        with TempNamedDir(tmp_path_dist_ckpt / "native_blockwise") as checkpoint_dir:
            native.save(state(weight), checkpoint_dir)()
            loaded = load(state(torch.empty_like(weight.dequantize())), checkpoint_dir)["w"]
            assert torch.equal(loaded, weight.dequantize())

    @staticmethod
    def _saved_and_loaded(checkpoint_dir, in_proj, **mlp_weights):
        args = parse_args(ignore_unknown_args=True)
        args.save_tokenizer_assets = False
        init_basic_mock_args(args, 2, 1)
        init_checkpointing_mock_args(args, checkpoint_dir)
        args.no_save_rng = args.no_load_rng = True
        with (
            mock.patch("megatron.training.checkpointing.get_args", new=lambda: args),
            mock.patch.object(type(in_proj), "dequantize", side_effect=AssertionError),
        ):
            save_checkpoint(1, [_Block(in_proj, **mlp_weights)], None, None, 0)
        loaded = _Block(torch.zeros(384, 64, dtype=torch.bfloat16, device="cuda"))
        with mock.patch("megatron.training.checkpointing.get_args", new=lambda: args):
            load_checkpoint([loaded], None, None)
        return loaded

    def test_mxfp8_weights_are_saved_packed_as_their_values(self, tmp_path_dist_ckpt, run_config):
        seed = 3 * parallel_state.get_tensor_model_parallel_rank()
        fc1, fc1_values = _packed("mxfp8", 256, 64, seed)
        fc2, fc2_values = _packed("mxfp8", 64, 128, seed + 1)
        in_proj, in_proj_values = _packed("mxfp8", 384, 64, seed + 2)
        with TempNamedDir(tmp_path_dist_ckpt / "native_mxfp8", sync=True) as checkpoint_dir:
            loaded = self._saved_and_loaded(checkpoint_dir, in_proj, fc1=fc1, fc2=fc2)
        assert torch.equal(loaded.mlp.linear_fc1.weight, fc1_values)
        assert torch.equal(loaded.mlp.linear_fc2.weight, fc2_values)
        assert torch.equal(loaded.in_proj, in_proj_values)

    def test_row_scaled_nvfp4_weights_are_saved_packed_as_their_values(
        self, tmp_path_dist_ckpt, run_config
    ):
        in_proj, values = _packed("nvfp4", 384, 64, parallel_state.get_tensor_model_parallel_rank())
        with TempNamedDir(tmp_path_dist_ckpt / "native_nvfp4", sync=True) as checkpoint_dir:
            loaded = self._saved_and_loaded(checkpoint_dir, in_proj)
        assert torch.equal(loaded.in_proj, values)

    def test_the_state_registered_after_a_load_is_the_state_of_the_next_save(
        self, tmp_path_dist_ckpt, run_config
    ):
        model, optimizer = setup_model_and_optimizer(1, 2, 1)
        args = parse_args(ignore_unknown_args=True)
        args.save_tokenizer_assets = False
        with (
            TempNamedDir(tmp_path_dist_ckpt / "native_after_load", sync=True) as checkpoint_dir,
            mock.patch("megatron.training.checkpointing.get_args", new=lambda: args),
            mock.patch("megatron.training.ckpt.native.get_args", new=lambda: args),
        ):
            init_basic_mock_args(args, 2, 1)
            init_checkpointing_mock_args(args, checkpoint_dir)
            save_checkpoint(1, model, optimizer, None, 0)
            native.spin_down()
            iteration, flops = load_checkpoint(model, optimizer, None, restore_training_state=True)
            native.register_after_load(model, optimizer, None, iteration, flops)
            session_id = native._session_id
            save_checkpoint(2, model, optimizer, None, 0)
            assert session_id is not None and native._session_id == session_id
