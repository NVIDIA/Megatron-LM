# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed-checkpoint coverage for SonicMoE parallel topologies."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, load_plain_tensors, save
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.moe.sonic_moe_layer import replace_moe_layer_specs_with_sonic_moe
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.dist_checkpointing.utils import setup_moe_model_and_optimizer
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("sonicmoe")

ATTENTION_TP = 2
EXPERT_TP = 1
NUM_EXPERTS = 8


def _initialize_sonic_moe_model(
    pre_process=True,
    post_process=True,
    seed=0,
    use_glu=True,
    use_sp=True,
    use_te=False,
    use_grouped_mlp=False,
    **config_kwargs,
):
    """Build a small SonicMoE GPT model and verify its TP group sizes."""
    config_kwargs.pop("pg_collection", None)
    config_kwargs.pop("config", None)
    config_kwargs["expert_tensor_parallel_size"] = EXPERT_TP
    assert use_glu
    assert use_sp
    assert not use_te

    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    config = TransformerConfig(
        num_layers=8,
        hidden_size=16,
        num_attention_heads=8,
        num_moe_experts=NUM_EXPERTS,
        use_cpu_initialization=True,
        sequence_parallel=use_sp,
        moe_grouped_gemm=use_grouped_mlp,
        add_bias_linear=False,
        gated_linear_unit=use_glu,
        **config_kwargs,
    )

    assert config.tensor_model_parallel_size == ATTENTION_TP
    assert config.expert_tensor_parallel_size == EXPERT_TP
    assert parallel_state.get_tensor_model_parallel_world_size() == ATTENTION_TP
    assert parallel_state.get_expert_tensor_parallel_world_size() == EXPERT_TP

    spec = get_gpt_layer_local_spec(num_experts=NUM_EXPERTS, moe_grouped_gemm=use_grouped_mlp)
    assert replace_moe_layer_specs_with_sonic_moe(spec) == 1
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=128,
        max_sequence_length=4,
        pre_process=pre_process,
        post_process=post_process,
    )
    model.bfloat16()
    with torch.no_grad():
        for param in model.parameters():
            param.random_()
    return model


class TestSonicMoECheckpoint:
    """Check Sonic model and optimizer resharding with combined PP and EP."""

    def teardown_method(self):
        """Destroy model-parallel groups after each topology test."""
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        ("src_tp_pp_ep", "dst_tp_pp_ep"),
        [
            pytest.param((2, 2, 2), (2, 2, 2), id="tp2-etp1-pp2-ep2"),
            pytest.param((2, 2, 2), (2, 1, 4), id="pp2-ep2-to-pp1-ep4"),
        ],
    )
    def test_model_and_optimizer_resharding(self, tmp_path_dist_ckpt, src_tp_pp_ep, dst_tp_pp_ep):
        """Reshard model weights and fully-reshardable optimizer states."""
        Utils.initialize_distributed()
        with (
            TempNamedDir(tmp_path_dist_ckpt / "sonic_moe_checkpoint_A", sync=True) as ckpt_a,
            TempNamedDir(tmp_path_dist_ckpt / "sonic_moe_checkpoint_B", sync=True) as ckpt_b,
        ):
            self._initialize_topology(src_tp_pp_ep)
            model_a, optimizer_a = self._build_model_and_optimizer(seed=2, topology=src_tp_pp_ep)
            metadata = {"distrib_optim_sharding_type": "fully_reshardable"}
            model_state_a = model_a[0].sharded_state_dict()
            state_a = {
                "model": model_state_a,
                "optimizer": optimizer_a.sharded_state_dict(model_state_a, metadata=metadata),
            }
            save(state_a, ckpt_a)
            Utils.destroy_model_parallel()

            self._initialize_topology(dst_tp_pp_ep)
            model_b, optimizer_b = self._build_model_and_optimizer(seed=3, topology=dst_tp_pp_ep)
            model_state_b = model_b[0].sharded_state_dict()
            load_template = {
                "model": model_state_b,
                "optimizer": optimizer_b.sharded_state_dict(
                    model_state_b, metadata=metadata, is_loading=True
                ),
            }
            loaded_state = load(load_template, ckpt_a)
            model_b[0].load_state_dict(loaded_state["model"])
            optimizer_b.load_state_dict(loaded_state["optimizer"])

            model_state_b = model_b[0].sharded_state_dict()
            state_b = {
                "model": model_state_b,
                "optimizer": optimizer_b.sharded_state_dict(model_state_b, metadata=metadata),
            }
            save(state_b, ckpt_b)
            Utils.destroy_model_parallel()

            Utils.initialize_model_parallel(1, 1)
            diffs = diff(load_plain_tensors(ckpt_a), load_plain_tensors(ckpt_b))
            assert not any(map(bool, diffs)), diffs

    @staticmethod
    def _initialize_topology(topology):
        """Initialize attention TP=2 and expert TP=1 for a TP/PP/EP topology."""
        tp, pp, ep = topology
        assert tp == ATTENTION_TP
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=EXPERT_TP,
        )
        assert parallel_state.get_tensor_model_parallel_world_size() == ATTENTION_TP
        assert parallel_state.get_expert_tensor_parallel_world_size() == EXPERT_TP

    @staticmethod
    def _build_model_and_optimizer(seed, topology):
        """Construct the Sonic model and distributed optimizer for a topology."""
        tp, pp, ep = topology
        return setup_moe_model_and_optimizer(
            seed=seed,
            tp=tp,
            pp=pp,
            ep=ep,
            initialize_fn=_initialize_sonic_moe_model,
            bf16=True,
            dist_opt=True,
            use_te=False,
            use_grouped_mlp=False,
            use_glu=True,
        )
