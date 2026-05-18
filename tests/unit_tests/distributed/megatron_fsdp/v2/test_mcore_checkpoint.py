# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
import logging
import shutil
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

import megatron.core.parallel_state as mpu
from megatron.core.utils import is_torch_min_version
from tests.unit_tests.distributed.megatron_fsdp.utils import (
    make_gpt_mock_data_iterator,
    make_moe_args_model_and_optimizer,
    pretrain_forward_backward,
    set_manual_seed,
)
from tests.unit_tests.test_utilities import Utils

logger = logging.getLogger(__name__)

SHARED_TMP_DIR = "/tmp/pytest-shared-tmp"


def _state_dict_to_full_tensor(sd):
    """Convert all DTensor values in a state dict to full (gathered) tensors."""
    from torch.distributed.tensor import DTensor

    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        uneven_dtensor_to_full_tensor,
    )

    out = {}
    for k, v in sd.items():
        if isinstance(v, DTensor):
            out[k] = uneven_dtensor_to_full_tensor(v)
        else:
            out[k] = v
    return out


def _normalize_key(key: str) -> str:
    """Strip all leading 'module.' prefixes to get the canonical parameter name."""
    while key.startswith("module."):
        key = key[len("module.") :]
    return key


def _build_key_mapping(source_sd, target_sd):
    """
    Build a mapping from source state dict keys to target state dict keys.

    Uses canonical (stripped) parameter names for matching.  Returns a
    dict suitable for passing to DCP.load where values are the target
    DTensor objects and keys match the checkpoint (source) format.
    """
    target_by_canonical = {}
    for t_key in target_sd:
        target_by_canonical[_normalize_key(t_key)] = target_sd[t_key]

    mapped = {"model": {}}
    for s_key in source_sd:
        canonical = _normalize_key(s_key)
        if canonical in target_by_canonical:
            mapped["model"][s_key] = target_by_canonical[canonical]
    return mapped


def _get_model_from_chunks(model_chunks):
    """Extract a single model from model_chunks (list or single module)."""
    if isinstance(model_chunks, list):
        return model_chunks[0]
    return model_chunks


class TestMegatronFsdpV2Checkpoint:
    """
    Megatron FSDP v2 checkpoint save/load and online format conversion tests.

    Covers:
    - Megatron FSDP v2 → Megatron FSDP v2 round-trip (save + load)
    - ND-parallel → Megatron FSDP v2 online conversion
    - Megatron FSDP v1 baseline → Megatron FSDP v2 online conversion
    """

    # ------------------------------------------------------------------
    # Model init / training helpers (same pattern as test_mcore_nd_parallel.py)
    # ------------------------------------------------------------------
    @staticmethod
    def _init_model_and_optimizer(seed=42, **kwargs):
        """
        Initialize model-parallel groups, build model and optimizer.

        Returns (model_chunks, optim).

        NOTE: Process groups are intentionally NOT destroyed here because
        the caller (e.g. DCP.load) may need them.  The caller is responsible
        for calling Utils.destroy_model_parallel() when done.
        """
        VOCAB_SIZE = kwargs.get("vocab_size", 100)
        MAX_SEQ_LEN = kwargs.get("seq_length", 128)
        MICRO_BATCH_SIZE = kwargs.get("micro_batch_size", 2)
        GLOBAL_BATCH_SIZE = kwargs.get("global_batch_size", 32)
        NUM_TRAINING_STEPS = kwargs.get("train_iters", 2)
        TP = kwargs.get("TP", 1)
        PP = kwargs.get("PP", 1)
        VPP = kwargs.get("VPP", None)
        EP = kwargs.get("EP", 1)
        ETP = kwargs.get("ETP", 1)
        OUTER_DP = kwargs.get("OUTER_DP", 1)

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=TP,
            pipeline_model_parallel_size=PP,
            expert_model_parallel_size=EP,
            expert_tensor_parallel_size=ETP,
            num_distributed_optimizer_instances=OUTER_DP,
        )

        set_manual_seed(seed)

        model_chunks, optim = make_moe_args_model_and_optimizer(
            ut_filename="test_mcore_checkpoint.py",
            micro_batch_size=MICRO_BATCH_SIZE,
            global_batch_size=GLOBAL_BATCH_SIZE,
            vocab_size=VOCAB_SIZE,
            padded_vocab_size=VOCAB_SIZE,
            seq_length=MAX_SEQ_LEN,
            max_position_embeddings=MAX_SEQ_LEN,
            sequence_parallel=TP > 1,
            tensor_model_parallel_size=TP,
            pipeline_model_parallel_size=PP,
            num_layers_per_virtual_pipeline_stage=VPP,
            train_iters=NUM_TRAINING_STEPS,
            **kwargs,
        )

        return model_chunks, optim

    @staticmethod
    def _training_loop(seed=42, **kwargs):
        """
        Run _init_model_and_optimizer followed by a deterministic training
        loop and return the model together with its state dict.

        If ``save`` is in kwargs, calls MCore's ``save_checkpoint`` at the
        end to persist the trained model.
        """
        model_chunks, optim = TestMegatronFsdpV2Checkpoint._init_model_and_optimizer(
            seed=seed, **kwargs
        )

        VOCAB_SIZE = kwargs.get("vocab_size", 100)
        MAX_SEQ_LEN = kwargs.get("seq_length", 128)
        MICRO_BATCH_SIZE = kwargs.get("micro_batch_size", 2)
        GLOBAL_BATCH_SIZE = kwargs.get("global_batch_size", 32)
        NUM_TRAINING_STEPS = kwargs.get("train_iters", 2)
        DP_GROUP = mpu.get_data_parallel_group()

        data_iterator = make_gpt_mock_data_iterator(
            dp_group=DP_GROUP,
            vocab_size=VOCAB_SIZE,
            sequence_length=MAX_SEQ_LEN,
            batch_size=MICRO_BATCH_SIZE,
            num_samples=GLOBAL_BATCH_SIZE * NUM_TRAINING_STEPS,
        )

        for _ in range(NUM_TRAINING_STEPS):
            optim.zero_grad()
            pretrain_forward_backward(
                model=model_chunks,
                data_iterator=data_iterator,
                sequence_length=MAX_SEQ_LEN,
                micro_batch_size=MICRO_BATCH_SIZE,
                num_micro_batches=GLOBAL_BATCH_SIZE // MICRO_BATCH_SIZE // DP_GROUP.size(),
            )
            optim.step()

        if "save" in kwargs:
            from megatron.training.checkpointing import save_checkpoint as mcore_save_checkpoint

            mcore_save_checkpoint(
                iteration=0,
                model=model_chunks,
                optimizer=optim,
                opt_param_scheduler=None,
                num_floating_point_operations_so_far=0,
            )

        model = _get_model_from_chunks(model_chunks)
        state_dict = model.state_dict()
        return model, state_dict

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    # ==================================================================
    # Test: Megatron FSDP v2 → Megatron FSDP v2 (round-trip)
    # ==================================================================
    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"),
        reason="Requires DTensor and DeviceMesh support (PyTorch >= 2.4.0).",
    )
    @pytest.mark.parametrize(
        "sharding_strategy",
        [
            pytest.param("optim_grads_params", id="optim_grads_params"),
            pytest.param("optim_grads", id="optim_grads"),
        ],
    )
    def test_megatron_fsdp_v2_round_trip(self, sharding_strategy):
        """
        Train a Megatron FSDP v2 model, save via MCore native
        ``save_checkpoint``, load via ``setup_model_and_optimizer``, and
        verify parameters match.
        """
        v2_config = dict(
            use_megatron_fsdp=True,
            use_fully_shard_api=True,
            init_model_with_meta_device=True,
            ckpt_format="fsdp_dtensor",
            gradient_accumulation_fusion=False,
            overlap_param_gather=True,
            overlap_grad_reduce=True,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
            data_parallel_sharding_strategy=sharding_strategy,
            fp8_param_gather=False,
        )

        ckpt_base = (
            Path(SHARED_TMP_DIR)
            / TestMegatronFsdpV2Checkpoint.__name__
            / f"v2_round_trip_{sharding_strategy}"
        )

        # ---- Train and save via MCore save_checkpoint ----
        # Passing save=<ckpt_base> triggers save_checkpoint inside _training_loop.
        save_config = dict(save=str(ckpt_base), save_interval=1, no_save_optim=True, no_save_rng=True, **v2_config)
        _, source_sd = TestMegatronFsdpV2Checkpoint._training_loop(**save_config)
        source_full = _state_dict_to_full_tensor(source_sd)
        Utils.destroy_model_parallel()

        # ---- Load via setup_model_and_optimizer's load_checkpoint ----
        load_config = dict(load=str(ckpt_base), no_load_optim=True, no_load_rng=True, **v2_config)
        v2_model_chunks, _ = TestMegatronFsdpV2Checkpoint._init_model_and_optimizer(
            **load_config,
        )
        v2_model = _get_model_from_chunks(v2_model_chunks)

        # ---- Verify ----
        loaded_full = _state_dict_to_full_tensor(v2_model.state_dict())
        nonempty = False
        for s_key, s_val in source_full.items():
            canonical = _normalize_key(s_key)
            matched_key = None
            for l_key in loaded_full:
                if _normalize_key(l_key) == canonical:
                    matched_key = l_key
                    break
            assert (
                matched_key is not None
            ), f"Key {s_key} (canonical: {canonical}) not found in loaded state dict"
            l_val = loaded_full[matched_key]
            if s_val is not None and s_val.numel() > 0:
                nonempty = True
            if s_val is not None and l_val is not None:
                assert (
                    s_val.shape == l_val.shape
                ), f"Shape mismatch for {s_key}: {s_val.shape} vs {l_val.shape}"
            else:
                assert s_val is None and l_val is None, f"One of source or loaded value for {s_key} is None while the other is not"
            assert_close(s_val, l_val, atol=0, rtol=0, msg=f"Value mismatch for {s_key}")

        world_size = torch.distributed.get_world_size()
        all_nonempty = [False] * world_size
        torch.distributed.all_gather_object(all_nonempty, nonempty)
        assert any(all_nonempty), "All ranks had empty model state after load."

        # Cleanup
        Utils.destroy_model_parallel()
        if torch.distributed.get_rank() == 0:
            shutil.rmtree(ckpt_base)
        torch.distributed.barrier()

    # ==================================================================
    # Test: ND-parallel → Megatron-FSDP v2
    # ==================================================================
    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"),
        reason="Requires DTensor and DeviceMesh support (PyTorch >= 2.4.0).",
    )
    @pytest.mark.parametrize("nd_topology", [pytest.param({"EP": 2}, id="EP2")])
    def test_nd_parallel_to_megatron_fsdp_v2(self, nd_topology):
        """
        Save a checkpoint from an ND-parallel (distributed-optimizer) model
        and load it into a Megatron-FSDP v2 model using
        ``MegatronFSDPStateful``.  Verify the state dict.
        """
        from megatron.core.distributed.fsdp.checkpoint import MegatronFSDPStateful
        from torch.distributed.checkpoint import load as dcp_load
        from torch.distributed.checkpoint import save as dcp_save

        nd_topology_str = "_".join([f"{k}{v}" for k, v in nd_topology.items()])

        # ---- ND-parallel: train and save ----
        source_model, source_sd = TestMegatronFsdpV2Checkpoint._training_loop(
            use_distributed_optimizer=True,
            data_parallel_sharding_strategy="optim_grads_params",
            fp8_param_gather=False,
            **nd_topology,
        )
        source_full = _state_dict_to_full_tensor(source_sd)

        ckpt_dir = (
            Path(SHARED_TMP_DIR)
            / TestMegatronFsdpV2Checkpoint.__name__
            / f"nd_parallel_{nd_topology_str}"
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
        dcp_save({"model": source_sd}, checkpoint_id=str(ckpt_dir))
        Utils.destroy_model_parallel()

        # ---- Megatron-FSDP v2: load via MegatronFSDPStateful ----
        v2_model_chunks, _ = TestMegatronFsdpV2Checkpoint._init_model_and_optimizer(
            use_megatron_fsdp=True,
            use_fully_shard_api=True,
            init_model_with_meta_device=True,
            ckpt_format="fsdp_dtensor",
            gradient_accumulation_fusion=False,
            overlap_param_gather=True,
            overlap_grad_reduce=True,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
            data_parallel_sharding_strategy="optim_grads_params",
            fp8_param_gather=False,
            **nd_topology,
        )
        v2_model = _get_model_from_chunks(v2_model_chunks)

        st = MegatronFSDPStateful(v2_model)
        state = {"model": st.state_dict()["model"]}
        mapped_sd = _build_key_mapping(source_sd, state["model"])
        dcp_load(state_dict=mapped_sd, checkpoint_id=str(ckpt_dir))

        # ---- Verify ----
        loaded_sd = v2_model.state_dict()
        loaded_full = _state_dict_to_full_tensor(loaded_sd)

        nonempty = False
        for s_key, s_val in source_full.items():
            canonical = _normalize_key(s_key)
            matched_key = None
            for l_key in loaded_full:
                if _normalize_key(l_key) == canonical:
                    matched_key = l_key
                    break
            assert (
                matched_key is not None
            ), f"Key {s_key} (canonical: {canonical}) not found in v2 state dict"
            l_val = loaded_full[matched_key]
            if s_val.numel() > 0:
                nonempty = True
            assert (
                s_val.shape == l_val.shape
            ), f"Shape mismatch for {s_key}: {s_val.shape} vs {l_val.shape}"
            assert_close(s_val, l_val, atol=0, rtol=0, msg=f"Value mismatch for {s_key}")

        world_size = torch.distributed.get_world_size()
        all_nonempty = [False] * world_size
        torch.distributed.all_gather_object(all_nonempty, nonempty)
        assert any(all_nonempty), "All ranks had empty model state after load."

        # Cleanup
        Utils.destroy_model_parallel()
        if torch.distributed.get_rank() == 0:
            shutil.rmtree(ckpt_dir)
        torch.distributed.barrier()

    # ==================================================================
    # Test: Megatron-FSDP baseline → Megatron-FSDP v2
    # ==================================================================
    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"),
        reason="Requires DTensor and DeviceMesh support (PyTorch >= 2.4.0).",
    )
    @pytest.mark.parametrize("nd_topology", [pytest.param({"EP": 2}, id="EP2")])
    @pytest.mark.parametrize(
        "source_configs",
        [
            pytest.param(
                dict(data_parallel_sharding_strategy="optim_grads_params"), id="optim_grads_params"
            ),
            pytest.param(dict(data_parallel_sharding_strategy="optim_grads"), id="optim_grads"),
            pytest.param(dict(data_parallel_sharding_strategy="optim"), id="optim"),
        ],
    )
    def test_megatron_fsdp_baseline_to_megatron_fsdp_v2(self, nd_topology, source_configs):
        """
        Save a checkpoint from a Megatron-FSDP baseline model and load it
        into a Megatron-FSDP v2 model using ``MegatronFSDPStateful``.
        Verify the state dict.
        """
        from megatron.core.distributed.fsdp.checkpoint import MegatronFSDPStateful
        from torch.distributed.checkpoint import load as dcp_load
        from torch.distributed.checkpoint import save as dcp_save

        nd_topology_str = "_".join([f"{k}{v}" for k, v in nd_topology.items()])
        shard_str = source_configs["data_parallel_sharding_strategy"]

        # ---- Megatron-FSDP baseline: train and save ----
        baseline_configs = copy.deepcopy(source_configs)
        baseline_configs.update(
            dict(
                use_megatron_fsdp=True,
                init_model_with_meta_device=True,
                ckpt_format="fsdp_dtensor",
                gradient_accumulation_fusion=False,
            )
        )
        source_model, source_sd = TestMegatronFsdpV2Checkpoint._training_loop(
            **nd_topology, **baseline_configs
        )
        source_full = _state_dict_to_full_tensor(source_sd)

        ckpt_dir = (
            Path(SHARED_TMP_DIR)
            / TestMegatronFsdpV2Checkpoint.__name__
            / f"baseline_{shard_str}_{nd_topology_str}"
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
        dcp_save({"model": source_sd}, checkpoint_id=str(ckpt_dir))
        Utils.destroy_model_parallel()

        # ---- Megatron-FSDP v2: load via MegatronFSDPStateful ----
        v2_configs = copy.deepcopy(source_configs)
        v2_configs.update(
            dict(
                use_megatron_fsdp=True,
                use_fully_shard_api=True,
                init_model_with_meta_device=True,
                ckpt_format="fsdp_dtensor",
                gradient_accumulation_fusion=False,
                overlap_param_gather=True,
                overlap_grad_reduce=True,
                recompute_granularity="full",
                recompute_method="uniform",
                recompute_num_layers=1,
            )
        )
        v2_model_chunks, _ = TestMegatronFsdpV2Checkpoint._init_model_and_optimizer(
            **nd_topology, **v2_configs
        )
        v2_model = _get_model_from_chunks(v2_model_chunks)

        st = MegatronFSDPStateful(v2_model)
        state = {"model": st.state_dict()["model"]}
        mapped_sd = _build_key_mapping(source_sd, state["model"])
        dcp_load(state_dict=mapped_sd, checkpoint_id=str(ckpt_dir))

        # ---- Verify ----
        loaded_sd = v2_model.state_dict()
        loaded_full = _state_dict_to_full_tensor(loaded_sd)

        nonempty = False
        for s_key, s_val in source_full.items():
            canonical = _normalize_key(s_key)
            matched_key = None
            for l_key in loaded_full:
                if _normalize_key(l_key) == canonical:
                    matched_key = l_key
                    break
            assert (
                matched_key is not None
            ), f"Key {s_key} (canonical: {canonical}) not found in v2 state dict"
            l_val = loaded_full[matched_key]
            if s_val.numel() > 0:
                nonempty = True
            assert (
                s_val.shape == l_val.shape
            ), f"Shape mismatch for {s_key}: {s_val.shape} vs {l_val.shape}"
            assert_close(s_val, l_val, atol=0, rtol=0, msg=f"Value mismatch for {s_key}")

        world_size = torch.distributed.get_world_size()
        all_nonempty = [False] * world_size
        torch.distributed.all_gather_object(all_nonempty, nonempty)
        assert any(all_nonempty), "All ranks had empty model state after load."

        # Cleanup
        Utils.destroy_model_parallel()
        if torch.distributed.get_rank() == 0:
            shutil.rmtree(ckpt_dir)
        torch.distributed.barrier()

