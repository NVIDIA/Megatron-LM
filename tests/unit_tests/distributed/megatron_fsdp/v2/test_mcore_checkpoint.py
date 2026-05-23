# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _state_dict_to_full_tensor(sd):
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
    while key.startswith("module."):
        key = key[len("module.") :]
    return key


def _get_model_from_chunks(model_chunks):
    if isinstance(model_chunks, list):
        return model_chunks[0]
    return model_chunks


def _assert_model_match(source_full, loaded_full):
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
        if s_val is None and l_val is None:
            continue
        assert (
            s_val is not None and l_val is not None
        ), f"One of source or loaded value for {s_key} is None while the other is not"
        assert (
            s_val.shape == l_val.shape
        ), f"Shape mismatch for {s_key}: {s_val.shape} vs {l_val.shape}"
        if s_val.numel() > 0:
            nonempty = True
        else:
            continue
        assert_close(
            s_val,
            l_val,
            rtol=0,
            atol=0,
            msg=f"Value mismatch for {s_key}, s_val: {s_val}, l_val: {l_val}",
        )

    world_size = torch.distributed.get_world_size()
    all_nonempty = [False] * world_size
    torch.distributed.all_gather_object(all_nonempty, nonempty)
    assert any(all_nonempty), "All ranks had empty model state after load."


def _optim_state_to_full(optim_sd_or_optim, model):
    """Wrap optimizer states as DTensors or unflatten flat format and gather to full tensors.

    Accepts either a sharded_state_dict (fsdp_dtensor format) or a
    DistributedOptimizer instance (nd fully_reshardable format).
    """
    from torch.distributed.tensor import DTensor

    from megatron.core.distributed.fsdp.checkpoint import _build_dtensor_optim_sd
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        uneven_dtensor_to_full_tensor,
    )

    # ND-parallel source: use get_parameter_state_dp_zero directly
    if not isinstance(optim_sd_or_optim, dict):
        return _nd_optim_state_to_full(optim_sd_or_optim, model)

    optim_sd = optim_sd_or_optim
    if "param_state_sharding_type" in optim_sd:
        return _flat_optim_state_to_full(optim_sd, model)

    wrapped = {"optimizer": _build_dtensor_optim_sd(optim_sd, model)}
    out = {}
    for param_name, param_states in wrapped["optimizer"]["state"].items():
        out[param_name] = {}
        for state_key, state_val in param_states.items():
            if isinstance(state_val, DTensor):
                out[param_name][state_key] = uneven_dtensor_to_full_tensor(state_val)
            else:
                out[param_name][state_key] = state_val
    return out


def _nd_optim_state_to_full(optim, model):
    """Get full optimizer state from an ND-parallel DistributedOptimizer.

    Uses ``get_parameter_state_dp_zero`` (the same method as
    ``sharded_param_state_fully_reshardable``) to gather all-gathered
    world tensors, then unflattens them per-parameter using
    ``param_index_map``.  Handles ``ChainedOptimizer`` (MoE) by
    iterating all inner ``DistributedOptimizer`` instances.
    """
    param_to_name = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            param_to_name[p] = name

    out = {}
    inner_optims = getattr(optim, "chained_optimizers", [optim])
    for inner_optim in inner_optims:
        dp_zero = inner_optim.get_parameter_state_dp_zero(
            use_gloo_comm=False, return_on_all_ranks=True
        )
        for gbuf_idx, gbuf_range_maps in enumerate(inner_optim.gbuf_ranges):
            buffer = inner_optim.buffers[gbuf_idx]
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                world_tensors = dp_zero[gbuf_idx][dtype]
                for model_param, (start, end, _) in buffer.param_index_map.items():
                    if model_param not in param_to_name:
                        continue
                    param_name = param_to_name[model_param]
                    out[param_name] = {
                        state_key: world_tensors[state_key][start:end]
                        .reshape(model_param.shape)
                        .clone()
                        .cuda()
                        for state_key in world_tensors
                        if state_key in ("param", "exp_avg", "exp_avg_sq")
                    }
    return out


def _flat_optim_state_to_full(optim_sd, model):
    """Convert fully_reshardable optimizer sd to {param_name: {state_key: tensor}}.

    The ``fully_reshardable`` format stores optimizer state as a nested
    dict: ``optim_sd["param_state"]`` = ``{0: {"param": ShardedTensor,
    "exp_avg": ShardedTensor, ...}, 1: {...}, ...}``.  This function
    extracts the full tensors from the ShardedTensor wrappers and maps
    integer indices to parameter names by matching the ``"param"``
    tensor shape against ``model.named_parameters()``.
    """
    from torch.distributed.checkpoint.stateful import ShardedTensor

    param_state = optim_sd.get("param_state", {})
    if not param_state:
        return {}

    # Build shape → param_name lookup
    param_by_shape = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            param_by_shape.setdefault(p.shape, []).append(name)

    out = {}
    used_names = set()
    for idx, states in sorted(param_state.items()):
        # Find matching param name via the "param" tensor shape
        param_tensor = None
        if "param" in states:
            st = states["param"]
            if isinstance(st, ShardedTensor):
                shards = st.local_shards()
                if shards:
                    param_tensor = shards[0].tensor
        param_name = None
        if param_tensor is not None:
            shape = param_tensor.shape
            candidates = [n for n in param_by_shape.get(shape, []) if n not in used_names]
            if candidates:
                param_name = candidates[0]
                used_names.add(param_name)
        if param_name is None:
            param_name = f"param_{idx}"

        out[param_name] = {}
        for state_key, st in states.items():
            if isinstance(st, ShardedTensor):
                shards = st.local_shards()
                out[param_name][state_key] = shards[0].tensor if shards else st
            else:
                out[param_name][state_key] = st

    return out


def _assert_optim_match(source_optim_full, loaded_optim_full, ignore_param=True):
    for param_name, source_states in source_optim_full.items():
        canonical = _normalize_key(param_name)
        matched_param = None
        for l_param in loaded_optim_full:
            if _normalize_key(l_param) == canonical:
                matched_param = l_param
                break
        assert matched_param is not None, (
            f"Optimizer param {param_name} (canonical: {canonical}) "
            f"not found in loaded optimizer state"
        )
        loaded_states = loaded_optim_full[matched_param]
        for state_key, s_val in source_states.items():
            if ignore_param and state_key == "param":
                continue
            assert (
                state_key in loaded_states
            ), f"Optimizer state '{state_key}' for param {param_name} not found after load"
            l_val = loaded_states[state_key]
            if s_val is None and l_val is None:
                continue
            if isinstance(s_val, torch.Tensor) and s_val.numel() > 0:
                assert (
                    s_val.shape == l_val.shape
                ), f"Optimizer state shape mismatch for {param_name}.{state_key}"
                assert_close(
                    s_val,
                    l_val,
                    atol=0,
                    rtol=0,
                    msg=f"Optimizer state value mismatch for {param_name}.{state_key}, s_val: {s_val}, l_val: {l_val}",
                )


# ==================================================================
# Test class
# ==================================================================


class TestMegatronFsdpV2Checkpoint:

    @staticmethod
    def _init_model_and_optimizer(seed=42, **kwargs):
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
        from megatron.training.training import get_optimizer_param_scheduler

        opt_param_scheduler = get_optimizer_param_scheduler(optim)
        return model_chunks, optim, opt_param_scheduler

    @staticmethod
    def _training_loop(seed=42, **kwargs):
        model_chunks, optim, scheduler = TestMegatronFsdpV2Checkpoint._init_model_and_optimizer(
            seed=seed, **kwargs
        )
        MICRO_BATCH_SIZE = kwargs.get("micro_batch_size", 2)
        GLOBAL_BATCH_SIZE = kwargs.get("global_batch_size", 32)
        MAX_SEQ_LEN = kwargs.get("seq_length", 128)
        DP_GROUP = mpu.get_data_parallel_group()

        data_iterator = make_gpt_mock_data_iterator(
            dp_group=DP_GROUP,
            vocab_size=kwargs.get("vocab_size", 100),
            sequence_length=MAX_SEQ_LEN,
            batch_size=MICRO_BATCH_SIZE,
            num_samples=GLOBAL_BATCH_SIZE * kwargs.get("train_iters", 2),
        )
        for _ in range(kwargs.get("train_iters", 2)):
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
                opt_param_scheduler=scheduler,
                num_floating_point_operations_so_far=0,
            )

        model = _get_model_from_chunks(model_chunks)
        return model, model.state_dict(), optim, scheduler

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    # ==================================================================
    # Online checkpoint conversion
    # ==================================================================

    # ---- Shared target config (always MFSDP v2) ----
    _TARGET_BASE = dict(
        use_megatron_fsdp=True,
        use_megatron_fsdp_v2=True,
        init_model_with_meta_device=True,
        ckpt_format="fsdp_dtensor",
        gradient_accumulation_fusion=False,
        overlap_param_gather=True,
        overlap_grad_reduce=True,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        fp8_param_gather=False,
    )

    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"),
        reason="Requires DTensor and DeviceMesh support (PyTorch >= 2.4.0).",
    )
    @pytest.mark.parametrize("nd_topology", [pytest.param({"EP": 2}, id="EP2")])
    @pytest.mark.parametrize(
        "source_type, source_configs, target_configs",
        [
            # ---- ND-parallel → MFSDP v2 ----
            pytest.param(
                "nd",
                dict(
                    distrib_optim_sharding_type="fully_reshardable",
                    dist_ckpt_optim_fully_reshardable=True,
                ),
                dict(data_parallel_sharding_strategy="optim_grads_params"),
                id="nd_fully_reshardable_to_v2",
            ),
            # ---- MFSDP v2 → MFSDP v2 (round-trip) ----
            pytest.param(
                "v2",
                dict(data_parallel_sharding_strategy="optim_grads_params"),
                dict(data_parallel_sharding_strategy="optim_grads_params"),
                id="v2_rt_optim_grads_params",
            ),
            pytest.param(
                "v2",
                dict(data_parallel_sharding_strategy="optim_grads"),
                dict(data_parallel_sharding_strategy="optim_grads"),
                id="v2_rt_optim_grads",
            ),
            # ---- MFSDP v2 → MFSDP v2 (cross-setting) ----
            pytest.param(
                "v2",
                dict(data_parallel_sharding_strategy="optim_grads_params"),
                dict(data_parallel_sharding_strategy="optim_grads"),
                id="v2_x_optim_grads_params_to_optim_grads",
            ),
            pytest.param(
                "v2",
                dict(data_parallel_sharding_strategy="optim_grads"),
                dict(data_parallel_sharding_strategy="optim_grads_params"),
                id="v2_x_optim_grads_to_optim_grads_params",
            ),
            # ---- MFSDP v1 baseline → MFSDP v2 ----
            pytest.param(
                "v1",
                dict(data_parallel_sharding_strategy="optim_grads_params"),
                dict(data_parallel_sharding_strategy="optim_grads_params"),
                id="v1_to_v2_optim_grads_params",
            ),
            pytest.param(
                "v1",
                dict(data_parallel_sharding_strategy="optim_grads"),
                dict(data_parallel_sharding_strategy="optim_grads"),
                id="v1_to_v2_optim_grads",
            ),
            pytest.param(
                "v1",
                dict(data_parallel_sharding_strategy="optim"),
                dict(data_parallel_sharding_strategy="optim_grads_params"),
                id="v1_optim_to_v2_optim_grads_params",
            ),
        ],
    )
    def test_checkpoint_online_convert(
        self, request, nd_topology, source_type, source_configs, target_configs
    ):
        """
        Train a source model (ND-parallel, MFSDP v1, or MFSDP v2) with
        *source_configs*, save via ``save_checkpoint``, load into an
        MFSDP v2 model with *target_configs*, and verify both model and
        optimizer state.
        """
        if source_type == "v1":
            pytest.skip("v1 checkpoint format not available")

        # ---- Build source config ----
        if source_type == "v2":
            src_base = dict(self._TARGET_BASE, auto_detect_ckpt_format=True)
        elif source_type == "v1":
            src_base = dict(
                use_megatron_fsdp=True,
                init_model_with_meta_device=True,
                ckpt_format="fsdp_dtensor",
                gradient_accumulation_fusion=False,
            )
        elif source_type == "nd":
            src_base = dict(
                use_distributed_optimizer=True, fp8_param_gather=False, ckpt_format="torch_dist"
            )
        else:
            raise ValueError(f"Unknown source_type: {source_type}")

        # ND-parallel → V2: optimizer format is incompatible (bucket-based vs name-based).
        src_sharding_type = source_configs.get("distrib_optim_sharding_type", "fsdp_dtensor")
        supports_optim = src_sharding_type != "dp_reshardable"

        save_config = {**nd_topology, **src_base, **source_configs}
        tgt_configs = {**nd_topology, **self._TARGET_BASE, **target_configs}
        if not supports_optim:
            save_config["no_save_optim"] = True
            tgt_configs["no_load_optim"] = True

        test_id = request.node.name.replace("[", "_").replace("]", "").replace("/", "_")
        ckpt_base = Path(SHARED_TMP_DIR) / TestMegatronFsdpV2Checkpoint.__name__ / test_id

        # ---- Train + save with source config ----
        source_model, source_sd, source_optim, _ = TestMegatronFsdpV2Checkpoint._training_loop(
            save=str(ckpt_base), save_interval=1, **save_config
        )
        source_full = _state_dict_to_full_tensor(source_sd)

        if supports_optim:
            if source_type == "nd":
                source_optim_full = _optim_state_to_full(source_optim, source_model)
            else:
                source_optim_sd = source_optim.sharded_state_dict(
                    model_sharded_state_dict=(
                        source_model.sharded_state_dict() if source_type not in ["v1", "v2"] else {}
                    ),
                    metadata={"distrib_optim_sharding_type": src_sharding_type},
                )
                source_optim_full = _optim_state_to_full(source_optim_sd, source_model)
        Utils.destroy_model_parallel()

        # ---- Load with target config (always MFSDP v2) ----
        v2_model_chunks, loaded_optim, _ = TestMegatronFsdpV2Checkpoint._init_model_and_optimizer(
            load=str(ckpt_base), **tgt_configs
        )
        v2_model = _get_model_from_chunks(v2_model_chunks)

        # ---- Verify model ----
        loaded_full = _state_dict_to_full_tensor(v2_model.state_dict())
        _assert_model_match(source_full, loaded_full)

        # ---- Verify optimizer (FSDP source types only) ----
        if supports_optim:
            loaded_optim_sd = loaded_optim.sharded_state_dict(
                metadata={"distrib_optim_sharding_type": "fsdp_dtensor"}
            )
            loaded_optim_full = _optim_state_to_full(loaded_optim_sd, v2_model)
            _assert_optim_match(source_optim_full, loaded_optim_full)

        Utils.destroy_model_parallel()
        if torch.distributed.get_rank() == 0:
            shutil.rmtree(ckpt_base, ignore_errors=True)
        torch.distributed.barrier()
