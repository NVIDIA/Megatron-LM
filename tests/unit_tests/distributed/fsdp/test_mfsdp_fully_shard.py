# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import transformer_engine as te
from packaging import version
from torch.nn.functional import mse_loss
from torch.optim import Adam

from tests.unit_tests.test_utilities import Utils

logger = logging.getLogger(__name__)

HSDP = "hsdp"
DP = "dp"
DP_SHARD = "dp_shard"
DP_OUTER = "dp_outer"
CP = "cp"
DP_SHARD_CP = "dp_shard_cp"
TP = "tp"
NO_SHARD = "no_shard"
OPTIM = "optim"
OPTIM_GRADS = "optim_grads"
OPTIM_GRADS_PARAMS = "optim_grads_params"
CNN = "cnn"
TRANSFORMER = "transformer"
TE_TRANSFORMER = "te_transformer"
DIM_SIZE = 2
NUM_LAYERS = 2
NUM_STEPS = 2

# Needed for `torch.distributed.checkpoint.{save,load}` because
# multiple processes need to write to the same directory.
SHARED_TMP_DIR = "/tmp/pytest-shared-tmp"


def destroy_device_mesh(device_mesh):

    # Teardown device mesh.
    del device_mesh
    try:
        from torch.distributed.device_mesh import _mesh_resources

        _mesh_resources.child_to_root_mapping.clear()
        _mesh_resources.root_to_flatten_mapping.clear()
        _mesh_resources.mesh_stack.clear()
        _mesh_resources.mesh_dim_group_options.clear()
        _mesh_resources.flatten_name_to_root_dims.clear()
    except Exception as e:
        # Global _MeshEnv is on a convoluted deprecation path.
        # Attempt to clean the global state, otherwise skip.
        logger.warning(f"Did not clean the deprecated DeviceMesh global state. Skipping...\n{e}")
        pass


class ToyCNN(torch.nn.Module):
    """Toy CNN model for testing Megatron-FSDP sharding for high-rank Tensor parameters and inputs."""

    def __init__(
        self,
        channels: int = 3,
        height: int = 10,
        width: int = 10,
        kernel_size: int = 3,
        output_dim: int = 10,
        bias: bool = True,
        num_layers: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.bias = bias
        self.num_layers = num_layers
        self.cnn_layers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(channels, channels, kernel_size, padding="same", bias=bias)
                for _ in range(num_layers)
            ]
        )
        self.dense = torch.nn.Linear(channels, 1, bias)

    def forward(self, x: torch.Tensor):
        """Toy forward pass for the CNN, where input and output shapes match."""
        x = x.broadcast_to(1, self.channels, self.height, self.width)
        for layer in self.cnn_layers:
            x = layer(x)
        x = x.transpose(1, 2).transpose(2, 3)
        x = self.dense(x).reshape(1, self.height, self.width)
        return x


class ToyTransformer(torch.nn.Module):
    """Toy Transformer model for testing Megatron-FSDP."""

    def __init__(self, model_dim, num_heads, num_layers, output_dim):
        super().__init__()
        self.transformer = torch.nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
        self.fc_out = torch.nn.Linear(model_dim, output_dim)

    def forward(self, x, y):
        x = self.transformer(x, y)
        x = self.fc_out(x)
        return x


class ToyTETransformer(torch.nn.Module):
    """Toy Transformer model for testing Megatron-FSDP with Transformer Engine."""

    def __init__(self, model_dim, num_heads, num_layers, output_dim):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                te.pytorch.TransformerLayer(
                    hidden_size=model_dim, ffn_hidden_size=model_dim, num_attention_heads=num_heads
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = te.pytorch.Linear(model_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.fc_out(x)
        return x


def build_toy_model(model_type: str, init_model_with_meta_device: bool, seed=None):
    """
    Helper function to build a toy model for testing Megatron-FSDP.
    """
    # Set the seed to make sure the same model is initialized on all ranks.
    if seed is not None:
        torch.manual_seed(seed)
    # Initialize on meta device or CUDA device. For CPU, use nullcontext() instead,
    # but for these tiny models we can just move everything to CUDA immediately.
    with torch.device("meta") if init_model_with_meta_device else torch.device("cuda"):
        if model_type == CNN:
            toy_model = ToyCNN(
                channels=3,
                height=DIM_SIZE,
                width=DIM_SIZE,
                kernel_size=3,
                output_dim=DIM_SIZE,
                bias=True,
                num_layers=NUM_LAYERS,
            )
            fsdp_unit_modules = [torch.nn.Conv2d, torch.nn.Linear]
        elif model_type == TRANSFORMER:
            toy_model = ToyTransformer(
                model_dim=DIM_SIZE, num_heads=2, num_layers=NUM_LAYERS, output_dim=DIM_SIZE
            )
            fsdp_unit_modules = [torch.nn.Transformer]
        elif model_type == TE_TRANSFORMER:
            toy_model = ToyTETransformer(
                model_dim=DIM_SIZE, num_heads=2, num_layers=NUM_LAYERS, output_dim=DIM_SIZE
            )
            fsdp_unit_modules = [te.pytorch.TransformerLayer]

    # Return the toy model, optimizer, and FSDP unit modules.
    return toy_model, fsdp_unit_modules


def build_distributed_environment(mesh_dim_config: tuple):
    """
    Helper function to build a distributed environment for testing Megatron-FSDP.
    Order of dimensions is (DP_OUTER, DP_SHARD, CP, TP).
    """
    from torch.distributed.device_mesh import init_device_mesh

    # Construct device mesh.
    device_mesh = init_device_mesh(
        "cuda", mesh_shape=mesh_dim_config, mesh_dim_names=(DP_OUTER, DP_SHARD, CP, TP)
    )
    # DP: Only relevant when using HSDP, where we need the flattened DP group for data parallelism. (Otherwise, just pass dp_shard.)
    device_mesh[(DP_OUTER, DP_SHARD)]._flatten(DP)
    # DP-Shard-CP: Only required if using CP. Otherwise, just pass dp_shard to FSDP.
    device_mesh[(DP_SHARD, CP)]._flatten(DP_SHARD_CP)
    # HSDP (DP-CP): Only required if using HSDP. Otherwise, don't pass hybrid_fsdp_group to Megatron-FSDP.
    device_mesh[(DP_OUTER, DP_SHARD, CP)]._flatten(HSDP)

    # Return the device mesh.
    return device_mesh


class TestMegatronFsdpFullyShard:
    """
    Test the fully_shard API for Megatron-FSDP.

    FIXME(@cspades): Megatron-FSDP leaves behind corrupted NCCL state that affects other tests.
    Until this is repaired, this test must be run in a separate bucket / container.
    """

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.4.0'),
        reason="Requires DTensor and DeviceMesh support in (approximately) PyTorch 2.4.0 or later. Should not be run on 2.2.0a0+81ea7a4 (LTS).",
    )
    @pytest.mark.parametrize("model_type", [CNN, TRANSFORMER, TE_TRANSFORMER])
    @pytest.mark.parametrize(
        "dp_shard_strategy", [NO_SHARD, OPTIM, OPTIM_GRADS, OPTIM_GRADS_PARAMS]
    )
    @pytest.mark.parametrize("dp_outer_strategy", [None, NO_SHARD, OPTIM])
    @pytest.mark.parametrize(
        "mesh_dim_config",
        [
            # (DP_OUTER, DP_SHARD, CP, TP)
            (2, 2, 2, 1),
            (1, 2, 2, 2),
            # TODO(@cspades, @boxiangw): Add a DTensor-based TP model
            # case to test strided sharding when using HSDP + TP.
            (2, 2, 1, 2),
        ],
    )
    @pytest.mark.parametrize("preserve_fp32_weights", [True, False])
    @pytest.mark.parametrize("init_model_with_meta_device", [True, False])
    def test_fully_shard(
        self,
        model_type,
        dp_shard_strategy,
        dp_outer_strategy,
        mesh_dim_config,
        preserve_fp32_weights,
        init_model_with_meta_device,
    ):
        """
        Test the fully_shard API with different configurations.
        Does NOT test for performance or convergence.

        NOTE(@cspades): This test is combinatorially large,
        don't add any new parameters unless absolutely necessary,
        or if some combinations can be flattened or simplified.
        """
        from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import fully_shard

        # Skip due to lack of functionality.
        if init_model_with_meta_device and dp_shard_strategy == NO_SHARD:
            pytest.skip(
                "Meta device initialization (init_model_with_meta_device=True) is not "
                "supported or necessary for the 'no_shard' / 0 sharding strategy."
            )
        elif dp_outer_strategy == OPTIM:
            if dp_shard_strategy != OPTIM_GRADS_PARAMS:
                # FIXME(@shjwudp, @cspades): This is an unexpected lack of support.
                # [default0]:FAILED tests/unit_tests/distributed/test_mfsdp_fully_shard.py
                # [False-True-True-True-mesh_dim_config0-optim-optim-cnn]
                # [False-True-True-True-mesh_dim_config0-optim-optim_grads-cnn]
                pytest.skip(
                    f"dp_outer sharding strategy {dp_outer_strategy} requires "
                    "zero_dp_strategy to be full-sharded ('optim_grads_params', 3)."
                )

        # Construct device mesh.
        device_mesh = build_distributed_environment(mesh_dim_config)

        # Construct toy model.
        toy_model, fsdp_unit_modules = build_toy_model(model_type, init_model_with_meta_device)
        toy_adam = Adam(params=toy_model.parameters(), lr=0.01)

        # Wrap in fully_shard.
        model, optimizer = fully_shard(
            module=toy_model,
            optimizer=toy_adam,
            device_mesh=device_mesh,
            dp_shard_dim=DP_SHARD_CP,
            dp_outer_dim=DP_OUTER if dp_outer_strategy is not None else None,
            tp_dim=TP,
            hybrid_fsdp_group=(
                device_mesh[HSDP].get_group() if dp_outer_strategy is not None else None
            ),
            fsdp_unit_modules=fsdp_unit_modules,
            zero_dp_strategy=dp_shard_strategy,
            outer_dp_sharding_strategy=(
                dp_outer_strategy if dp_outer_strategy is not None else "no_shard"
            ),
            preserve_fp32_weights=preserve_fp32_weights,
            grad_reduce_in_fp32=False,
            init_model_with_meta_device=init_model_with_meta_device,
        )

        # Mock input and target.
        toy_input = torch.randn(1, DIM_SIZE, DIM_SIZE).to("cuda")
        toy_target = torch.randn(1, DIM_SIZE, DIM_SIZE).to("cuda")

        for step in range(NUM_STEPS):
            # Synchronize model parameters and gradients on the final training step only.
            if step == NUM_STEPS - 1:
                model.set_model_auto_sync(True)
            else:
                model.set_model_auto_sync(False)

            # Forward pass.
            if model_type == CNN or model_type == TE_TRANSFORMER:
                output = model(toy_input)
            elif model_type == TRANSFORMER:
                output = model(toy_input, toy_input)

            # Loss.
            loss = mse_loss(output, toy_target)

            # Backward pass.
            loss.backward()

            # Validate gradients exist in the Torch Module, i.e. non-None and non-zero.
            grads_exist = any(
                isinstance(p.grad, torch.Tensor) and p.grad.to_local().count_nonzero().item() > 0
                for p in model.parameters()
            )
            sharding_group = (
                device_mesh[HSDP].get_group()
                if dp_outer_strategy == OPTIM
                else device_mesh[DP_SHARD_CP].get_group()
            )
            if dp_shard_strategy != NO_SHARD:
                # Because of uneven sharding, we need to gather the result from all ranks
                # to verify if any gradients exist or not at this step of training.
                grads_exist_gathered = [None] * sharding_group.size()
                torch.distributed.all_gather_object(
                    object_list=grads_exist_gathered, obj=grads_exist, group=sharding_group
                )
                # Gradients exist on at least one of the optimizer sharding ranks.
                grads_exist = any(grads_exist_gathered)

            # Gradients do not exist until synchronization is activated.
            if step == NUM_STEPS - 1:
                assert grads_exist, "Root module gradients should exist on final microbatch."
            else:
                assert (
                    not grads_exist
                ), "Root module gradients should not exist prior to optimization step."
            torch.distributed.barrier()

            # Optimizer step. Apply accumulated gradients to the model weights.
            if step == NUM_STEPS - 1:
                optimizer.step()
                optimizer.zero_grad()

        # Required to reset the parallelism environment.
        destroy_device_mesh(device_mesh)

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.4.0'),
        reason="Requires DTensor and DeviceMesh support in (approximately) PyTorch 2.4.0 or later. Should not be run on 2.2.0a0+81ea7a4 (LTS).",
    )
    @pytest.mark.parametrize("shard_strategy", [OPTIM_GRADS_PARAMS, OPTIM_GRADS, OPTIM, NO_SHARD])
    @pytest.mark.parametrize("outer_shard_strategy", [NO_SHARD, OPTIM])
    @pytest.mark.parametrize("model_type", [CNN, TRANSFORMER, TE_TRANSFORMER])
    @pytest.mark.parametrize("mesh_dim_config", [(1, 4, 2, 1), (2, 2, 2, 1)])
    def test_dcp_checkpoint_save_and_load(
        self, mesh_dim_config, shard_strategy, outer_shard_strategy, model_type
    ):
        """
        Test that an Megatron-FSDP model checkpoint can be saved and loaded accurately.
        """
        from torch.distributed.tensor import DTensor

        from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import fully_shard

        # Skip tests.
        if outer_shard_strategy == OPTIM and shard_strategy != OPTIM_GRADS_PARAMS:
            # FIXME(@shjwudp, @cspades): This is an unexpected lack of support.
            # [default0]:FAILED tests/unit_tests/distributed/test_mfsdp_fully_shard.py
            # [False-True-True-True-mesh_dim_config0-optim-optim-cnn]
            # [False-True-True-True-mesh_dim_config0-optim-optim_grads-cnn]
            pytest.skip(
                f"dp_outer sharding strategy {outer_shard_strategy} requires "
                "zero_dp_strategy to be full-sharded ('optim_grads_params', 3)."
            )
        if shard_strategy == NO_SHARD:
            # NOTE: Just directly checkpoint the MegatronFSDP.module.state_dict() using torch.save().
            # Beyond the scope of this unit test.
            pytest.xfail(reason="Megatron-FSDP does not support NO_SHARD for checkpointing yet.")

        """
        DISTRIBUTED ENVIRONMENT INIT
        """
        # Construct device mesh.
        device_mesh = build_distributed_environment(mesh_dim_config)

        """
        MODEL TRAINING

        Run through a single training step to update the model weights so the checkpoint
        accuracy tests are non-trivial, i.e. don't just use the initialized weights.
        """
        # Test model.
        toy_model, fsdp_unit_modules = build_toy_model(model_type, False, seed=0)
        toy_adam = Adam(params=toy_model.parameters(), lr=0.01)

        # Wrap in fully_shard.
        model, optimizer = fully_shard(
            module=toy_model,
            optimizer=toy_adam,
            device_mesh=device_mesh,
            dp_shard_dim=DP_SHARD_CP,
            dp_outer_dim=DP_OUTER,
            tp_dim=TP,
            hybrid_fsdp_group=device_mesh[HSDP].get_group(),
            fsdp_unit_modules=fsdp_unit_modules,
            zero_dp_strategy=shard_strategy,
            outer_dp_sharding_strategy=outer_shard_strategy,
            preserve_fp32_weights=True,
            grad_reduce_in_fp32=True,
            init_model_with_meta_device=False,
            sync_model_each_microbatch=True,
        )

        # Mock input and target.
        toy_input = torch.randn(1, DIM_SIZE, DIM_SIZE).to("cuda")
        toy_target = torch.randn(1, DIM_SIZE, DIM_SIZE).to("cuda")

        # Forward pass.
        if model_type == CNN or model_type == TE_TRANSFORMER:
            output = model(toy_input)
        elif model_type == TRANSFORMER:
            output = model(toy_input, toy_input)

        # Loss.
        loss = mse_loss(output, toy_target)

        # Backward pass.
        loss.backward()

        # Optimizer step.
        optimizer.step()
        optimizer.zero_grad()

        """
        MODEL PRE-SAVE CHECKPOINT VALUES
        """
        # Compute one more forward pass using the optimized model
        # weights to get a pre-save checkpoint validation loss.
        model.eval()
        if model_type == CNN or model_type == TE_TRANSFORMER:
            pre_output = model(toy_input)
        elif model_type == TRANSFORMER:
            pre_output = model(toy_input, toy_input)
        pre_save_loss = mse_loss(pre_output, toy_target).item()

        # Save deep copy of the model state dictionary before checkpointing.
        s1 = deepcopy(model.state_dict())

        # Save deep copy of the optimizer state dictionary before checkpointing.
        o1 = deepcopy(optimizer.state_dict())

        """
        MODEL CHECKPOINT SAVE
        """
        # Write model to checkpoint.
        CKPT_DIR = (
            Path(SHARED_TMP_DIR)
            / TestMegatronFsdpFullyShard.__name__
            / self.test_dcp_checkpoint_save_and_load.__name__
            / f"checkpoint_shard-{shard_strategy}_outer-{outer_shard_strategy}_{model_type}"
        )
        CKPT_DIR.mkdir(parents=True, exist_ok=True, mode=0o777)
        torch.distributed.checkpoint.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            checkpoint_id=str(CKPT_DIR),
        )

        """
        MODEL CHECKPOINT LOAD
        """
        # Initialize a new model for checkpoint loading. Set a different seed to force a different model init,
        # to ensure the checkpoint loading is accurate and non-trivial.
        toy_model, fsdp_unit_modules = build_toy_model(model_type, False, seed=1)
        toy_adam = Adam(params=toy_model.parameters(), lr=0.01)

        # Wrap in fully_shard.
        model, optimizer = fully_shard(
            module=toy_model,
            optimizer=toy_adam,
            device_mesh=device_mesh,
            dp_shard_dim=DP_SHARD_CP,
            dp_outer_dim=DP_OUTER,
            tp_dim=TP,
            hybrid_fsdp_group=device_mesh[HSDP].get_group(),
            fsdp_unit_modules=fsdp_unit_modules,
            zero_dp_strategy=shard_strategy,
            outer_dp_sharding_strategy=outer_shard_strategy,
            preserve_fp32_weights=True,
            grad_reduce_in_fp32=True,
            init_model_with_meta_device=False,
            sync_model_each_microbatch=True,
        )

        # Load model from checkpoint.
        ckpt_state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.distributed.checkpoint.load(state_dict=ckpt_state_dict, checkpoint_id=str(CKPT_DIR))
        model.load_state_dict(ckpt_state_dict["model"], strict=False)
        optimizer.load_state_dict(ckpt_state_dict["optimizer"])
        s2 = deepcopy(model.state_dict())
        o2 = deepcopy(optimizer.state_dict())

        """
        MODEL CHECKPOINT STATE DICT VALIDATION
        """
        # Compare pre-save and post-load model state dictionaries.
        for key in s1.keys() | s2.keys():
            v1 = s1.get(key, None)
            if isinstance(v1, DTensor):
                v1 = v1.to_local()
            v2 = s2.get(key, None)
            if isinstance(v2, DTensor):
                v2 = v2.to_local()
            assert (
                v1 is not None and v2 is not None
            ), f"[{key} Not Found] Original Param: {v1} | Checkpoint Param: {v2}"
            assert (
                v1.shape == v2.shape
            ), f"[Checkpoint Param {key} Shape Mismatch] {v1.shape} != {v2.shape}"
            assert torch.allclose(v1, v2), f"[Checkpoint Param {key} Value Mismatch] {v1} != {v2}"

        # Compare pre-save and post-load optimizer state dictionaries.
        for param_id in o1["state"].keys() | o2["state"].keys():
            param_state_1 = o1["state"].get(param_id, None)
            param_state_2 = o2["state"].get(param_id, None)
            assert (
                param_state_1 is not None and param_state_2 is not None
            ), f"[{param_id} Not Found] Original Param State: {param_state_1} | Checkpoint Param State: {param_state_2}"
            for key in param_state_1.keys() | param_state_2.keys():
                v1 = param_state_1.get(key, None)
                if isinstance(v1, DTensor):
                    v1 = v1.to_local()
                v2 = param_state_2.get(key, None)
                if isinstance(v2, DTensor):
                    v2 = v2.to_local()
                assert (
                    v1 is not None and v2 is not None
                ), f"[{param_id} {key} Not Found] Original Optimizer State: {v1} | Checkpoint Optimizer State: {v2}"
                assert (
                    v1.shape == v2.shape
                ), f"[Optimizer State {param_id} {key} Shape Mismatch] {v1.shape} != {v2.shape}"
                assert torch.allclose(
                    v1, v2
                ), f"[Optimizer State {param_id} {key} Value Mismatch] {v1} != {v2}"
        assert len(o1["param_groups"]) == len(
            o2["param_groups"]
        ), f"[Optimizer State Param Groups Length Mismatch] {o1['param_groups']} != {o2['param_groups']}"
        for i in range(len(o2["param_groups"])):
            for key in o1["param_groups"][i].keys():
                v1 = o1["param_groups"][i][key]
                v2 = o2["param_groups"][i][key]
                assert (
                    v1 == v2
                ), f"[Optimizer State Param Group {i} {key} Value Mismatch] {v1} != {v2}"

        """
        MODEL CHECKPOINT FORWARD PASS VALIDATION
        """
        # Forward pass using the post-load checkpoint model weights.
        model.eval()
        if model_type == CNN or model_type == TE_TRANSFORMER:
            post_output = model(toy_input)
        elif model_type == TRANSFORMER:
            post_output = model(toy_input, toy_input)
        post_load_loss = mse_loss(post_output, toy_target)

        # Validate the pre-save and post-load loss.
        assert (
            pre_save_loss == post_load_loss.item()
        ), f"[Rank {torch.distributed.get_rank()}] Pre-Save Loss: {pre_save_loss} != Post-Load Loss: {post_load_loss}"

        # Continue training.
        post_load_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        """
        CLEANUP
        """
        # Clean up temporary checkpoint directory.
        if torch.distributed.get_rank() == 0:
            shutil.rmtree(CKPT_DIR)
        torch.distributed.barrier()

        # Destroy device mesh.
        destroy_device_mesh(device_mesh)

    @pytest.mark.parametrize("shard_strategy", [OPTIM_GRADS_PARAMS, OPTIM_GRADS, OPTIM, NO_SHARD])
    def test_fully_shard_ez(self, shard_strategy):
        """
        Test fully_shard(device_mesh=None). Represents the easiest entrypoint to Megatron-FSDP.
        """
        from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
            fully_shard_model,
            fully_shard_optimizer,
        )

        # Construct toy model.
        toy_model, fsdp_unit_modules = build_toy_model(TRANSFORMER, False)

        # Fully-shard the model.
        mfsdp_model = fully_shard_model(
            module=toy_model, fsdp_unit_modules=fsdp_unit_modules, zero_dp_strategy=shard_strategy
        )

        # Initialize the distributed optimizer on the MegatronFSDP model.
        toy_adam = Adam(params=mfsdp_model.parameters(), lr=0.01)
        optimizer = fully_shard_optimizer(optimizer=toy_adam)

        # Mock input and target.
        toy_input = torch.randn(1, DIM_SIZE, DIM_SIZE).to("cuda")
        toy_target = torch.randn(1, DIM_SIZE, DIM_SIZE).to("cuda")

        for step in range(NUM_STEPS):

            # Forward pass.
            output = mfsdp_model(toy_input, toy_input)

            # Loss.
            loss = mse_loss(output, toy_target)

            # Backward pass.
            loss.backward()

            # Optimizer step.
            optimizer.step()
            optimizer.zero_grad()
