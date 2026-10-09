# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Expert scalar parameters keep their own Adam sharding with either Muon layout."""

import pytest
import torch

import megatron.core.optimizer as optimizer_module
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.training.training import wrap_model_chunks_with_ddp
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200


class _ExpertBiasModel(torch.nn.Module):
    """Expose dense and expert weights/biases without a token-dispatch dependency."""

    def __init__(self, expert_rank: int) -> None:
        super().__init__()
        # Biases span every DP shard even with the usual eight-rank test launcher.
        self.dense = torch.nn.Linear(16, 1024, device="cuda", dtype=torch.bfloat16)
        self.experts = torch.nn.Linear(16, 1024, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            self.dense.weight.fill_(0.125)
            self.dense.bias.fill_(0.5)
            self.experts.weight.fill_(0.25)
            self.experts.bias.fill_(expert_rank + 1)
        for param in self.dense.parameters():
            param.allreduce = True
        for param in self.experts.parameters():
            param.allreduce = False


@pytest.mark.skipif(
    not optimizer_module.HAVE_EMERGING_OPTIMIZERS, reason="requires emerging-optimizers"
)
@pytest.mark.parametrize("use_param_layout", [False, True])
@pytest.mark.parametrize("overlap_param_gather", [False, True])
def test_layerwise_expert_adam_sharding_and_updates(use_param_layout, overlap_param_gather):
    """Check real EP=2 reduce-scatter, Adam state updates and parameter publication.

    Main's compact layout kept expert biases inside LayerWise. Routing the scalar
    fallback to DistOpt must preserve expert-DP ownership instead of rejecting the
    bias or mixing different experts in a dense-DP collective.
    """
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
    )
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        assert pg.dp_cp.size() >= 4, "requires at least four ranks with EP=2"
        model = _ExpertBiasModel(pg.ep.rank())
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=1,
            num_moe_experts=2,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        ddp = wrap_model_chunks_with_ddp(
            [model],
            config,
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=True, overlap_param_gather=overlap_param_gather
            ),
            use_layer_wise_distributed_optimizer=True,
            use_layer_wise_param_layout=use_param_layout,
            pg_collection=pg,
        )[0]
        optimizer_config = OptimizerConfig(
            optimizer="muon",
            lr=0.01,
            weight_decay=0.0,
            bf16=True,
            clip_grad=0.0,
            muon_split_qkv=False,
            use_layer_wise_distributed_optimizer=True,
            overlap_param_gather=overlap_param_gather,
        )
        optimizer = get_megatron_optimizer(
            optimizer_config, [ddp], pg_collection=pg, use_gloo_process_groups=False
        )
        assert isinstance(optimizer.chained_optimizers[0], LayerWiseDistributedOptimizer)
        adam_optimizers = optimizer.chained_optimizers[1:]
        assert len(adam_optimizers) == 2
        assert all(isinstance(child, DistributedOptimizer) for child in adam_optimizers)
        biases = {"dense": model.dense.bias, "expert": model.experts.bias}
        expected_groups = {"dense": pg.dp_cp, "expert": pg.expt_dp}
        by_domain = {}
        for child in adam_optimizers:
            parameters = {param for buffer in child.buffers for param in buffer.params}
            domain = "expert" if model.experts.bias in parameters else "dense"
            assert parameters == {biases[domain]}
            assert child.data_parallel_group is expected_groups[domain]
            assert all(
                buffer.data_parallel_group is expected_groups[domain] for buffer in child.buffers
            )
            expected_buffers = ddp.expert_parallel_buffers if domain == "expert" else ddp.buffers
            assert all(buffer in expected_buffers for buffer in child.buffers)
            model_parallel_group = (
                getattr(pg, "tp_ep_pp_with_egtp_remat", pg.tp_ep_pp)
                if domain == "expert"
                else pg.mp
            )
            assert child.data_parallel_group_idx == model_parallel_group.rank()
            by_domain[domain] = child
        assert set(by_domain) == set(biases)

        references = {
            domain: torch.nn.Parameter(param.detach().float().clone())
            for domain, param in biases.items()
        }
        # Compare sharded and full updates through the same Adam kernel. TE and
        # PyTorch differ in scalar rounding, which is unrelated to DP ownership.
        reference_optimizer = optimizer_module.Adam(
            list(references.values()),
            lr=optimizer_config.lr,
            betas=(optimizer_config.adam_beta1, optimizer_config.adam_beta2),
            eps=optimizer_config.adam_eps,
            weight_decay=0.0,
        )
        for step in range(2):
            ddp.zero_grad_buffer()
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            local_gradient = (torch.distributed.get_rank() + 1) * (1.0 if step == 0 else -0.5)
            for param in model.parameters():
                param.main_grad.fill_(local_gradient)
            expected_grads = {}
            for domain, param in biases.items():
                expected_grad = torch.full_like(param, local_gradient, dtype=torch.float32)
                torch.distributed.all_reduce(expected_grad, group=expected_groups[domain])
                expected_grad /= pg.dp_cp.size()
                expected_grads[domain] = expected_grad
                references[domain].grad = expected_grad

            ddp.finish_grad_sync()
            assert not optimizer.prepare_grads()
            for domain, child in by_domain.items():
                for model_group, main_group in zip(
                    child.model_float16_groups, child.shard_fp32_from_float16_groups
                ):
                    for param, master in zip(model_group, main_group):
                        interval = child._get_model_param_range_map(param)["param"]
                        expected_grad = expected_grads[domain][interval.start : interval.end]
                        torch.testing.assert_close(master.grad, expected_grad, rtol=0, atol=0)

            reference_optimizer.step()
            assert optimizer.step_with_ready_grads()
            ddp.start_param_sync(force_sync=True)
            for domain, child in by_domain.items():
                full_master = torch.zeros_like(references[domain])
                for model_group, main_group in zip(
                    child.model_float16_groups, child.shard_fp32_from_float16_groups
                ):
                    for param, master in zip(model_group, main_group):
                        interval = child._get_model_param_range_map(param)["param"]
                        reference = references[domain][interval.start : interval.end]
                        torch.testing.assert_close(master, reference, rtol=0, atol=0)
                        for key in ("exp_avg", "exp_avg_sq"):
                            expected_state = reference_optimizer.state[references[domain]][key]
                            torch.testing.assert_close(
                                child.optimizer.state[master][key],
                                expected_state[interval.start : interval.end],
                                rtol=0,
                                atol=0,
                            )
                        full_master[interval.start : interval.end].copy_(master)
                torch.distributed.all_reduce(full_master, group=expected_groups[domain])
                torch.testing.assert_close(
                    biases[domain], full_master.to(torch.bfloat16), rtol=0, atol=0
                )
    finally:
        Utils.destroy_model_parallel()
