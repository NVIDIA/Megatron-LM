# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exercise the conversion checkpoint envelope using real distributed checkpoint I/O."""

from pathlib import Path

import pytest
import torch
import yaml

from megatron.bridge.models.conversion.checkpoint import (
    load_megatron_model,
    load_model_config,
    resolve_checkpoint_path,
    save_megatron_model,
)
from megatron.bridge.models.hybrid.model_config import BridgeHybridModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.training.config.yaml_utils import safe_yaml_representers
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(
    "config_format",
    ["flat-training", "flat-inference", "root-training", "root-inference", "root-both"],
)
def test_load_model_config_accepts_flat_and_root_checkpoint_metadata(
    tmp_path, monkeypatch, config_format
):
    """Conversion reads old configs and the training/inference wrapper from #9031."""
    config = BridgeHybridModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=2),
        hybrid_layer_pattern="**",
        vocab_size=128,
        seq_length=64,
    )
    container_module = "megatron.training.config.container"
    training = {
        "_target_": f"{container_module}.PretrainConfigContainer",
        "model": config.as_dict(),
    }
    inference = {
        "_target_": f"{container_module}.InferenceConfigContainer",
        "model": config.as_dict(),
    }
    if config_format == "flat-training":
        metadata = training
    elif config_format == "flat-inference":
        metadata = inference
    else:
        metadata = {"_target_": f"{container_module}.RootConfigContainer"}
        if config_format in ("root-training", "root-both"):
            metadata["training"] = training
        if config_format in ("root-inference", "root-both"):
            metadata["inference"] = inference
        if config_format == "root-both":
            # Match MLM's training-first selection, not the rollout model's metadata.
            inference["model"]["vocab_size"] = 256

    with safe_yaml_representers():
        (tmp_path / "run_config.yaml").write_text(yaml.safe_dump(metadata))
    # This test only needs metadata, not a model-weight checkpoint directory.
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.checkpoint.resolve_checkpoint_path",
        lambda path: tmp_path,
    )

    restored, _ = load_model_config(tmp_path)

    assert type(restored) is BridgeHybridModelConfig
    assert restored.vocab_size == 128
    assert restored.seq_length == 64
    assert restored.transformer.hidden_size == 16


@pytest.mark.parametrize("release", [False, True])
def test_auto_config_uses_tracker_not_newer_unpublished_directory(tmp_path_dist_ckpt, release):
    from megatron.bridge import AutoBridge
    from tests.unit_tests.bridge.qwen_vl.test_qwen35_vl_bridge import _qwen35_vl_config

    Utils.initialize_model_parallel(1, 1)
    try:
        reference = _qwen35_vl_config(moe=False)
        config = AutoBridge.from_hf_config(reference).get_model_config()
        config.vocab_size = 256
        with TempNamedDir(tmp_path_dist_ckpt / "tracked-config") as directory:
            directory = Path(directory)
            save_megatron_model([SmallModel(config)], directory)
            if torch.distributed.get_rank() == 0:
                reference.save_pretrained(directory / "reference")
                (directory / "iter_0000001").mkdir()
                if release:
                    (directory / "iter_0000000").rename(directory / "release")
                    (directory / "latest_checkpointed_iteration.txt").write_text("release")
            torch.distributed.barrier()
            restored = AutoBridge.from_auto_config(str(directory), str(directory / "reference"))
            assert restored.hf_pretrained.text_config.vocab_size == 256
    finally:
        Utils.destroy_model_parallel()


class SmallModel(torch.nn.Module):
    """Keep checkpoint testing independent of the model family's expensive kernels."""

    def __init__(self, config):
        super().__init__()
        self.model_config = config
        self.weight = torch.nn.Parameter(torch.zeros(4, 4, device="cuda"))

    def sharded_state_dict(self, metadata=None):
        return {
            "weight": ShardedTensor.from_rank_offsets(
                "weight", self.weight, replica_id=(0, 0, torch.distributed.get_rank())
            )
        }


class SmallBuilder:
    def get_model(self, config, *, load_weights, wrap_with_ddp):
        assert load_weights is False
        assert wrap_with_ddp is False
        return [SmallModel(config)]


@pytest.mark.parametrize(("saved_pp", "load_pp"), [(1, 1), (2, 2), (2, 1)])
def test_native_weights_and_model_config_round_trip_without_training_args(
    tmp_path_dist_ckpt, saved_pp, load_pp
):
    Utils.initialize_model_parallel(1, 1)
    try:
        config = BridgeHybridModelConfig(
            transformer=TransformerConfig(
                num_layers=2,
                hidden_size=16,
                num_attention_heads=2,
                pipeline_model_parallel_size=saved_pp,
                pipeline_model_parallel_layout=(
                    [["embedding", "decoder"], ["decoder", "loss"]] if saved_pp == 2 else None
                ),
            ),
            hybrid_layer_pattern="*|*" if saved_pp == 2 else "**",
            vocab_size=128,
            seq_length=64,
        )
        model = SmallModel(config)
        with torch.no_grad():
            model.weight.copy_(torch.arange(16, device="cuda").reshape(4, 4))

        with TempNamedDir(tmp_path_dist_ckpt / "bridge-roundtrip") as directory:
            save_megatron_model([model], directory)
            checkpoint = resolve_checkpoint_path(directory)
            restored_config, _ = load_model_config(checkpoint)
            assert type(restored_config) is type(config)
            assert restored_config.vocab_size == 128
            assert restored_config.seq_length == 64
            overrides = {"pipeline_model_parallel_size": load_pp}
            if saved_pp == load_pp == 2:
                # The export launcher supplies both the outer pattern and the nested layout.
                overrides["hybrid_layer_pattern"] = "*|*"
                overrides["pipeline_model_parallel_layout"] = config.pipeline_model_parallel_layout
            [restored] = load_megatron_model(SmallBuilder(), directory, mp_overrides=overrides)
            assert torch.equal(restored.weight, model.weight)
            if load_pp == 1:
                assert restored.model_config.pipeline_model_parallel_layout is None
                assert restored.model_config.hybrid_layer_pattern == "**"
            else:
                assert restored.model_config.hybrid_layer_pattern == "*|*"

            # Publishing into an existing iteration must not replace a prior export.
            with pytest.raises(RuntimeError, match="FileExistsError"):
                save_megatron_model([model], directory)
    finally:
        Utils.destroy_model_parallel()
