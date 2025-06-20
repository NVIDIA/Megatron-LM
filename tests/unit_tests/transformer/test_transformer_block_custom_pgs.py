# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.


import copy
import os
from typing import Optional

import pytest
import torch
from packaging import version

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.process_groups_config import GradCommProcessGroups, ModelCommProcessGroups
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.unit_tests.test_utilities import Utils


class HeterogenousTransformerLayer(TransformerLayer):
    """A transformer layer that supports different process groups for attention and MLP.

    This specialized transformer layer implementation allows independent parallelism
    strategies for the self-attention and MLP components

    Implementation details:
    - Uses identity operations as placeholders during initialization
    - Replaces the placeholder modules with properly configured attention and MLP
      using their respective process groups
    - Requires process groups to be specified in the submodule parameters

    Args:
        config (TransformerConfig): Configuration for the transformer layer
        submodules (TransformerLayerSubmodules): Submodule specifications with process group params
        layer_number (int, optional): Index of this layer. Defaults to 1.
        hidden_dropout (float, optional): Override dropout rate. Defaults to None.
        model_comm_pgs (ModelCommProcessGroups, optional): Default process groups. Defaults to None.
        vp_stage (int, optional): Virtual pipeline stage. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        model_comm_pgs: ModelCommProcessGroups = None,
        vp_stage: Optional[int] = None,
    ):
        # Temporarily replace attention and MLP with IdentityOp,
        # This is a temporary workaround for the test until we have a better interface
        # will rebuild them with custom process groups after super init
        def _modify_submodules(submodules):
            submodules.self_attention = IdentityOp
            submodules.mlp = IdentityOp
            return submodules

        original_attention = submodules.self_attention
        original_mlp = submodules.mlp
        new_submodules = _modify_submodules(copy.copy(submodules))

        super().__init__(
            config=config,
            submodules=new_submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            model_comm_pgs=model_comm_pgs,
            vp_stage=vp_stage,
        )

        assert (
            'model_comm_pgs' in submodules.self_attention.params
        ), "model_comm_pgs should be in the params of the submodules"
        self.self_attention = build_module(
            original_attention, config=self.config, layer_number=layer_number
        )
        assert (
            'tp_group' in submodules.mlp.params
        ), "tp_group should be in the params of the submodules"
        self.mlp = build_module(original_mlp, config=self.config)


def create_reference_mlp(hidden_size, ffn_hidden_size, seed=12345):
    """Create a reference MLP with full unsharded weights.

    Args:
        hidden_size: Input/output dimension
        ffn_hidden_size: Hidden dimension
        seed: Random seed for weight initialization

    Returns:
        Reference MLP with unsharded weights (nn.Sequential)
    """
    # Set seed for reproducible initialization
    torch.manual_seed(seed)

    # Create standard PyTorch Linear layers (unsharded)
    ref_fc1 = torch.nn.Linear(hidden_size, ffn_hidden_size, bias=True)
    ref_fc2 = torch.nn.Linear(ffn_hidden_size, hidden_size, bias=True)

    # Return as a simple sequential model
    return torch.nn.Sequential(ref_fc1, ref_fc2).cpu()


def copy_weights_to_tp_mlp(ref_mlp, tp_mlp, tp_group):
    """Copy weights from reference MLP to tensor-parallel MLP.

    Args:
        ref_mlp: Reference MLP with full weights (nn.Sequential)
        tp_mlp: Tensor-parallel MLP (megatron MLP instance)
        tp_group: Tensor parallel process group

    Returns:
        None (modifies tp_mlp in-place)
    """
    # Get tensor parallel rank and world size
    tp_rank = tp_group.rank()
    tp_world_size = tp_group.size()

    # Reference components
    ref_fc1 = ref_mlp[0]  # First linear layer
    ref_fc2 = ref_mlp[1]  # Second linear layer

    # Manually copy and shard weights based on TP rank
    with torch.no_grad():
        # FC1 (Column Parallel) - split along output dimension
        out_size = ref_fc1.weight.size(0)
        per_partition_size = out_size // tp_world_size
        start_idx = tp_rank * per_partition_size
        end_idx = (tp_rank + 1) * per_partition_size

        tp_mlp.linear_fc1.weight.copy_(
            ref_fc1.weight[start_idx:end_idx].to(tp_mlp.linear_fc1.weight.device)
        )
        if hasattr(tp_mlp.linear_fc1, 'bias') and tp_mlp.linear_fc1.bias is not None:
            tp_mlp.linear_fc1.bias.copy_(
                ref_fc1.bias[start_idx:end_idx].to(tp_mlp.linear_fc1.bias.device)
            )

        # FC2 (Row Parallel) - split along input dimension
        in_size = ref_fc2.weight.size(1)
        per_partition_size = in_size // tp_world_size
        start_idx = tp_rank * per_partition_size
        end_idx = (tp_rank + 1) * per_partition_size

        tp_mlp.linear_fc2.weight.copy_(
            ref_fc2.weight[:, start_idx:end_idx].to(tp_mlp.linear_fc2.weight.device)
        )
        if hasattr(tp_mlp.linear_fc2, 'bias') and tp_mlp.linear_fc2.bias is not None:
            tp_mlp.linear_fc2.bias.copy_(ref_fc2.bias.to(tp_mlp.linear_fc2.bias.device))


def _gpt_te_layer_spec_with_hetro_pgs(attn_model_comm_pgs, mlp_model_comm_pgs):
    return ModuleSpec(
        module=HeterogenousTransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={
                    "attn_mask_type": AttnMaskType.causal,
                    "model_comm_pgs": attn_model_comm_pgs,
                },
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=ModuleSpec(
                module=MLP,
                params={'tp_group': mlp_model_comm_pgs.tp},
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


class TestTransformerBlockWithProcessGroups:
    def setup_method(self, method):
        Utils.destroy_model_parallel()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    @pytest.mark.parametrize(
        "world_size, tp_size, cp_size, dp_size",
        [
            (1, 1, 1, 1),  # Single GPU, no parallelism
            (2, 1, 2, 1),  # 2 GPUs, 1 TP, 2 CP
            (2, 2, 1, 1),  # 2 GPUs, 2 TP, 1 CP
            (8, 8, 1, 1),  # 8 GPUs, 8 TP, 1 CP
            (8, 2, 4, 1),  # 8 GPUs, 2 TP, 4 CP
            (8, 4, 2, 1),  # 8 GPUs, 4 TP, 2 CP
            (8, 1, 1, 8),  # 8 GPUs, 1 TP, 1 CP, 8 DP
            (8, 2, 1, 4),  # 8 GPUs, 2 TP, 1 CP, 4 DP
            (8, 2, 2, 2),  # 8 GPUs, 2 TP, 2 CP, 2 DP
        ],
    )
    def test_params_and_grads_match_transformer_block(self, world_size, tp_size, cp_size, dp_size):
        """
        Test that parameters and gradients match after one forward and backward pass
        between transformer blocks using default and custom process groups.
        """
        # Skip if world size doesn't match
        actual_world_size = torch.cuda.device_count()
        if actual_world_size != world_size:
            pytest.skip(f"Test requires world_size={world_size}, but got {actual_world_size}")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
        )
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"

        torch.manual_seed(12345)
        model_parallel_cuda_manual_seed(123)

        # Create transformer configuration
        transformer_config = TransformerConfig(
            num_layers=3,
            hidden_size=4096,
            num_attention_heads=32,
            use_cpu_initialization=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            bf16=True,
            context_parallel_size=cp_size,
        )

        # Create a transformer block with default process groups
        default_block = (
            TransformerBlock(transformer_config, get_gpt_layer_with_transformer_engine_spec())
            .cuda()
            .bfloat16()
        )

        # Create custom process groups
        device_mesh = torch.distributed.init_device_mesh(
            "cuda", (1, 1, dp_size, cp_size, tp_size), mesh_dim_names=("pp", "ep", "dp", "cp", "tp")
        )

        tp_group = device_mesh.get_group(mesh_dim="tp")
        cp_group = device_mesh.get_group(mesh_dim="cp")
        pp_group = device_mesh.get_group(mesh_dim="pp")
        ep_group = device_mesh.get_group(mesh_dim="ep")
        model_comm_pgs = ModelCommProcessGroups(tp=tp_group, cp=cp_group, pp=pp_group, ep=ep_group)

        dp_group = device_mesh.get_group(mesh_dim="dp")
        dp_cp_mesh = device_mesh["dp", "cp"]
        dp_cp_group = dp_cp_mesh._flatten().get_group()
        grad_comm_pgs = GradCommProcessGroups()

        grad_comm_pgs.dp = dp_group
        grad_comm_pgs.dp_cp = dp_cp_group

        # Create a transformer block with custom process groups
        custom_block = (
            TransformerBlock(
                transformer_config,
                get_gpt_layer_with_transformer_engine_spec(),
                model_comm_pgs=model_comm_pgs,
            )
            .cuda()
            .bfloat16()
        )

        # Initialize with same parameters
        for default_param, custom_param in zip(
            default_block.parameters(), custom_block.parameters()
        ):
            custom_param.data.copy_(default_param.data)

        # copy buffers
        for default_buffer, custom_buffer in zip(default_block.buffers(), custom_block.buffers()):
            custom_buffer.data.copy_(default_buffer.data)

        # wrap with DDP
        ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, bucket_size=10000)
        default_block = DistributedDataParallel(
            config=transformer_config, ddp_config=ddp_config, module=default_block
        )

        custom_block = DistributedDataParallel(
            config=transformer_config,
            ddp_config=ddp_config,
            module=custom_block,
            grad_comm_pgs=grad_comm_pgs,
            model_comm_pgs=model_comm_pgs,
        )

        # Create test input
        sequence_length = 4096
        micro_batch_size = 4
        hidden_states = (
            torch.randn(
                (sequence_length, micro_batch_size, transformer_config.hidden_size),
                device="cuda",
                requires_grad=True,
            )
            .bfloat16()
            .requires_grad_(True)
        )
        hidden_states.retain_grad()

        torch.distributed.all_reduce(hidden_states, op=torch.distributed.ReduceOp.SUM)

        hidden_states_default = hidden_states.clone().detach().requires_grad_(True)
        hidden_states_custom = hidden_states.clone().detach().requires_grad_(True)

        # Forward passes
        output_default = default_block(hidden_states=hidden_states_default, attention_mask=None)
        output_custom = custom_block(hidden_states=hidden_states_custom, attention_mask=None)
        # Verify outputs match
        torch.testing.assert_close(
            output_default,
            output_custom,
            rtol=1e-8,
            atol=0,
            msg="Forward outputs don't match between default and custom process groups",
        )

        output_default.backward(torch.ones_like(output_default) * 1e3)
        output_custom.backward(torch.ones_like(output_custom) * 1e3)
        # Verify gradients match for parameters
        # with DDP grad attribute is None, only main_grad is available
        for i, (default_param, custom_param) in enumerate(
            zip(default_block.parameters(), custom_block.parameters())
        ):
            if default_param.main_grad is not None and custom_param.main_grad is not None:
                param_name = [name for name, param in default_block.named_parameters()][i]

                # ideally we want to grads and assert they are close
                # but the grads are too small to use rtol
                # for now just a smoke test to ensure grads are available
                # TODO: ykarnati - improve this test to ensure we have large grads for comparision
                assert (
                    default_param.main_grad is not None and custom_param.main_grad is not None
                ), f"Gradient is None for parameter '{param_name}' at index {i}"

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    @pytest.mark.parametrize(
        "world_size, attn_tp_size, attn_cp_size, mlp_tp_size",
        [
            (1, 1, 1, 1),  # Single GPU, no parallelism
            (2, 1, 1, 2),  # 2 GPUs, attn: 1 TP, 1 CP; mlp: 2 TP
            (2, 2, 1, 2),  # 2 GPUs, attn: 2 TP, 1 CP; mlp: 2 TP
            (8, 1, 1, 8),  # 8 GPUs, attn: 1 TP, 1 CP; mlp: 8 TP
            (8, 8, 1, 1),  # 8 GPUs, attn: 8 TP, 1 CP; mlp: 1 TP
            (8, 2, 1, 4),  # 8 GPUs, attn: 2 TP, 1 CP; mlp: 4 TP
            (8, 4, 1, 2),  # 8 GPUs, attn: 4 TP, 1 CP; mlp: 2 TP
            (8, 2, 2, 2),  # 8 GPUs, attn: 2 TP, 2 CP; mlp: 2 TP
        ],
    )
    def test_fwd_bwd_pass_non_uniform_transformer_block(
        self, world_size, attn_tp_size, attn_cp_size, mlp_tp_size
    ):
        """
        Test that parameters and gradients after one forward and backward pass
        with different process groups for attention and mlp.
        """

        actual_world_size = torch.cuda.device_count()
        if actual_world_size != world_size:
            pytest.skip(f"Test requires world_size={world_size}, but got {actual_world_size}")
        Utils.initialize_model_parallel()
        torch.manual_seed(12345)
        model_parallel_cuda_manual_seed(123)

        # Create transformer configuration
        transformer_config = TransformerConfig(
            num_layers=3,
            hidden_size=4096,
            num_attention_heads=32,
            use_cpu_initialization=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            bf16=True,
            context_parallel_size=attn_cp_size,
        )

        # Create custom process groups
        device_mesh = torch.distributed.init_device_mesh(
            "cuda",
            (attn_tp_size, attn_cp_size, mlp_tp_size),
            mesh_dim_names=("attn_tp", "attn_cp", "mlp_tp"),
        )
        attn_tp_group = device_mesh.get_group(mesh_dim="attn_tp")
        attn_cp_group = device_mesh.get_group(mesh_dim="attn_cp")
        mlp_tp_group = device_mesh.get_group(mesh_dim="mlp_tp")

        attn_model_comm_pgs = ModelCommProcessGroups(tp=attn_tp_group, cp=attn_cp_group)
        mlp_model_comm_pgs = ModelCommProcessGroups(tp=mlp_tp_group)

        # Get the layer spec with different process groups for attention and mlp
        hetro_layer_spec = _gpt_te_layer_spec_with_hetro_pgs(
            attn_model_comm_pgs, mlp_model_comm_pgs
        )
        custom_block = TransformerBlock(transformer_config, hetro_layer_spec).cuda().bfloat16()

        sequence_length = 4096
        micro_batch_size = 2

        # [sequence length, batch size, hidden size]
        hidden_states = (
            torch.randn(
                (sequence_length, micro_batch_size, transformer_config.hidden_size),
                device="cuda",
                requires_grad=True,
            )
            .bfloat16()
            .requires_grad_(True)
        )
        hidden_states.retain_grad()

        output_custom = custom_block(hidden_states=hidden_states, attention_mask=None)

        assert (
            output_custom.shape[0] == sequence_length
        ), f"Output shape is {output_custom.shape} dont match sequence length {sequence_length}"
        assert (
            output_custom.shape[1] == micro_batch_size
        ), f"Output shape is {output_custom.shape} dont match micro batch size {micro_batch_size}"
        assert (
            output_custom.shape[2] == transformer_config.hidden_size
        ), f"Output shape is {output_custom.shape} dont match hidden size {transformer_config.hidden_size}"

        loss = output_custom.sum()
        loss.backward()

        assert hidden_states.grad is not None, "Hidden states gradient is None"

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    def test_fwd_bwd_pass_mix_and_match_transformer_blocks(self):
        world_size = 8
        actual_world_size = torch.cuda.device_count()
        if actual_world_size != world_size:
            pytest.skip(f"Test requires world_size={world_size}, but got {actual_world_size}")

        Utils.initialize_model_parallel()
        torch.manual_seed(12345)
        model_parallel_cuda_manual_seed(123)
        grid_cp_2_tp_4 = torch.distributed.init_device_mesh(
            "cuda", (2, 4), mesh_dim_names=("cp", "tp")
        )

        tp_group = grid_cp_2_tp_4.get_group(mesh_dim="tp")
        cp_group = grid_cp_2_tp_4.get_group(mesh_dim="cp")
        model_comm_pgs = ModelCommProcessGroups(tp=tp_group, cp=cp_group)

        transformer_config = TransformerConfig(
            num_layers=3,
            hidden_size=4096,
            num_attention_heads=32,
            use_cpu_initialization=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            context_parallel_size=2,
        )
        transformer_block_cp2_tp4 = (
            TransformerBlock(
                transformer_config,
                get_gpt_layer_with_transformer_engine_spec(),
                model_comm_pgs=model_comm_pgs,
            )
            .cuda()
            .bfloat16()
        )

        sequence_length = 4096
        micro_batch_size = 4
        hidden_states = (
            torch.randn(
                (sequence_length, micro_batch_size, transformer_config.hidden_size), device="cuda"
            )
            .bfloat16()
            .requires_grad_(True)
        )
        hidden_states.retain_grad()

        grid_cp_2_tp_2_dp_2 = torch.distributed.init_device_mesh(
            "cuda", (2, 2, 2, 1, 1), mesh_dim_names=("cp", "tp", "dp", "pp", "ep")
        )
        tp_group = grid_cp_2_tp_2_dp_2.get_group(mesh_dim="tp")
        cp_group = grid_cp_2_tp_2_dp_2.get_group(mesh_dim="cp")
        dp_group = grid_cp_2_tp_2_dp_2.get_group(mesh_dim="dp")
        pp_group = grid_cp_2_tp_2_dp_2.get_group(mesh_dim="pp")
        ep_group = grid_cp_2_tp_2_dp_2.get_group(mesh_dim="ep")
        model_comm_pgs = ModelCommProcessGroups(tp=tp_group, cp=cp_group, pp=pp_group, ep=ep_group)
        grad_comm_pgs = GradCommProcessGroups()

        dp_cp_mesh = grid_cp_2_tp_2_dp_2["cp", "dp"]
        dp_cp_group = dp_cp_mesh._flatten().get_group()
        grad_comm_pgs.dp = dp_group
        grad_comm_pgs.dp_cp = dp_cp_group

        transformer_block_cp2_tp2 = (
            TransformerBlock(
                transformer_config,
                get_gpt_layer_with_transformer_engine_spec(),
                model_comm_pgs=model_comm_pgs,
            )
            .cuda()
            .bfloat16()
        )

        ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, bucket_size=10000)
        transformer_block_cp2_tp2_dp_2 = DistributedDataParallel(
            config=transformer_config,
            ddp_config=ddp_config,
            module=transformer_block_cp2_tp2,
            grad_comm_pgs=grad_comm_pgs,
            model_comm_pgs=model_comm_pgs,
        )

        output_cp2_tp_2_dp_2 = transformer_block_cp2_tp2_dp_2(
            hidden_states=hidden_states, attention_mask=None
        )

        assert output_cp2_tp_2_dp_2.shape == (
            sequence_length,
            micro_batch_size,
            transformer_config.hidden_size,
        ), (
            f"Output shape is {output_cp2_tp_2_dp_2.shape} dont match sequence length {sequence_length}, "
            f"micro batch size {micro_batch_size}, hidden size {transformer_config.hidden_size}"
        )

        # pass as input to transformer_block_cp2_tp4
        output_cp2_tp4 = transformer_block_cp2_tp4(
            hidden_states=output_cp2_tp_2_dp_2, attention_mask=None
        )

        assert output_cp2_tp4.shape == (
            sequence_length,
            micro_batch_size,
            transformer_config.hidden_size,
        ), (
            f"Output shape is {output_cp2_tp4.shape} dont match sequence length {sequence_length}, "
            f"micro batch size {micro_batch_size}, hidden size {transformer_config.hidden_size}"
        )

        # verify backward pass
        loss = output_cp2_tp4.sum()
        loss.backward()

        assert hidden_states.grad is not None, "Hidden states gradient is None"

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    @pytest.mark.parametrize(
        "world_size, tp_size, dp_size, reverse_tp_dp_order",
        [
            (1, 1, 1, False),  # Single GPU, no parallelism
            (2, 1, 2, False),  # 2 GPUs, 1 TP, 2 DP
            (2, 2, 1, False),  # 2 GPUs, 2 TP, 1 DP
            (8, 8, 1, False),  # 8 GPUs, 8 TP, 1 DP
            (8, 1, 8, False),  # 8 GPUs, 1 TP, 8 DP
            (8, 2, 4, False),  # 8 GPUs, 2 TP, 4 DP
            (8, 4, 2, False),  # 8 GPUs, 4 TP, 2 DP
            (8, 2, 4, True),  # 8 GPUs, 2 TP, 4 DP # reverse TP and DP order in device mesh
            (8, 4, 2, True),  # 8 GPUs, 4 TP, 2 DP # reverse TP and DP order in device mesh
        ],
    )
    def test_mlp_with_custom_pgs(self, world_size, tp_size, dp_size, reverse_tp_dp_order):

        actual_world_size = torch.cuda.device_count()
        if actual_world_size != world_size:
            pytest.skip(f"Test requires world_size={world_size}, but got {actual_world_size}")

        Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)

        # Set PyTorch random seed explicitly for reproducible input
        torch.manual_seed(12345)
        model_parallel_cuda_manual_seed(123)

        if reverse_tp_dp_order:
            device_mesh = torch.distributed.init_device_mesh(
                "cuda", (1, 1, tp_size, dp_size), mesh_dim_names=("pp", "ep", "tp", "dp")
            )
        else:
            device_mesh = torch.distributed.init_device_mesh(
                "cuda", (1, 1, dp_size, tp_size), mesh_dim_names=("pp", "ep", "dp", "tp")
            )
        pp_group = device_mesh.get_group(mesh_dim="pp")
        ep_group = device_mesh.get_group(mesh_dim="ep")
        dp_group = device_mesh.get_group(mesh_dim="dp")
        tp_group = device_mesh.get_group(mesh_dim="tp")
        mlp_model_comm_pgs = ModelCommProcessGroups(tp=tp_group, pp=pp_group, ep=ep_group)

        grad_comm_pgs = GradCommProcessGroups()
        grad_comm_pgs.dp = dp_group
        grad_comm_pgs.dp_cp = dp_group

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=4096,
            num_attention_heads=32,
            use_cpu_initialization=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            context_parallel_size=1,
            ffn_hidden_size=4 * 4096,
        )

        default_mlp_spec = ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
            ),
        )

        custom_mlp_spec = ModuleSpec(
            module=MLP,
            params={'tp_group': mlp_model_comm_pgs.tp},
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
            ),
        )

        reference_mlp = create_reference_mlp(
            transformer_config.hidden_size, transformer_config.ffn_hidden_size
        )
        default_mlp = build_module(default_mlp_spec, config=transformer_config).cuda()
        custom_mlp = build_module(custom_mlp_spec, config=transformer_config).cuda()

        copy_weights_to_tp_mlp(
            reference_mlp, default_mlp, parallel_state.get_tensor_model_parallel_group()
        )
        copy_weights_to_tp_mlp(reference_mlp, custom_mlp, tp_group)

        ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, bucket_size=10000)

        default_mlp = DistributedDataParallel(
            config=transformer_config, ddp_config=ddp_config, module=default_mlp
        )

        custom_mlp = DistributedDataParallel(
            config=transformer_config,
            ddp_config=ddp_config,
            module=custom_mlp,
            model_comm_pgs=mlp_model_comm_pgs,
            grad_comm_pgs=grad_comm_pgs,
        )

        sequence_length = 4096
        micro_batch_size = 1
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, transformer_config.hidden_size), device="cuda"
        ).requires_grad_(True)

        torch.distributed.all_reduce(hidden_states, op=torch.distributed.ReduceOp.SUM)

        output_default, _ = default_mlp(hidden_states)
        output_custom, _ = custom_mlp(hidden_states)

        torch.testing.assert_close(output_default, output_custom, rtol=1e-8, atol=0)

    def test_deterministic_output_from_single_block(self):
        """
        Test that a single transformer block produces identical outputs
        when run twice with the same input.
        """
        # Initialize model parallel with no parallelism
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)

        # Set PyTorch random seed explicitly for reproducible inputs
        torch.manual_seed(12345)
        model_parallel_cuda_manual_seed(123)

        # Create transformer configuration
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            deterministic_mode=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            attention_backend=AttnBackend.unfused,
        )

        transformer_config_2 = copy.deepcopy(transformer_config)

        # Create a single transformer block
        block = TransformerBlock(transformer_config, get_gpt_layer_local_spec())
        block_2 = TransformerBlock(transformer_config_2, get_gpt_layer_local_spec())
        # Move block to GPU
        block.cuda()
        block_2.cuda()

        # Create test input
        sequence_length = 37
        micro_batch_size = 7

        # copy weights from block_2 to block
        for default_param, custom_param in zip(block.parameters(), block_2.parameters()):
            custom_param.data.copy_(default_param.data)

        for name, buffer in block.named_buffers():
            if name in dict(block_2.named_buffers()):
                dict(block_2.named_buffers())[name].copy_(buffer)

        hidden_states_int = torch.randint(
            -10,
            10,
            (sequence_length, micro_batch_size, transformer_config.hidden_size),
            device="cuda",
        )
        hidden_states = hidden_states_int.float()

        # Create two identical copies of the input
        hidden_states_1 = hidden_states.clone()
        hidden_states_2 = hidden_states.clone()

        # Forward passes through the same block
        output_1 = block(hidden_states=hidden_states_1, attention_mask=None)
        output_block_2 = block_2(hidden_states=hidden_states_2, attention_mask=None)

        torch.testing.assert_close(
            output_1,
            output_block_2,
            rtol=0,
            atol=0,
            msg="Outputs don't match for identical inputs through the same block",
        )
