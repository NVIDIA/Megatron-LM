# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Flextron Utilities

Provides setup and configuration functions for Flextron elasticity.
Extracted from HybridModel to keep the core model clean.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron.core import mpu, parallel_state
from megatron.elastification.arguments import convert_per_lists_to_int_lists
from megatron.elastification.flextron_config import inject_flextron_config
from megatron.elastification.flextron_elasticity_hooks import apply_flextron_elasticity_to_model
from megatron.elastification.memory_config import MemoryConfig, load_memory_config
from megatron.elastification.router.flex_budget_utils import (
    get_memory_footprint,
    get_num_parameters,
)
from megatron.elastification.router.hybrid_flex_router import FlextronRouter
from megatron.training import get_args


class FlextronModelManager:
    """
    Manages Flextron functionality for a model.
    Handles router, budget calculations, and loss functions.
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        inject_flextron_config(get_args(), config)
        convert_per_lists_to_int_lists(config)
        config.hybrid_layer_pattern = getattr(model, 'hybrid_layer_pattern', '')
        self.router = None
        self.budget_type = getattr(config, 'budget_type', 'param')

        # Load memory quantization profile from args
        args = get_args()
        self.memory_config = load_memory_config(args)

        # Budget calculation attributes
        self.all_param = None
        self.total_memory = None

        # Hook managers
        self.hook_managers = []

    def setup_router(self):
        """Initialize the Flextron router if enabled."""
        if getattr(self.config, 'enable_router', False):  # and self.model.pre_process:
            self.router = FlextronRouter(config=self.config)

            # Make router name pipeline-stage-aware to avoid naming conflicts in PP>1
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            router_name = f"router_pp{pp_rank}"

            # Set the router with pipeline-specific name
            setattr(self.model, router_name, self.router)
            self.model.router = self.router
        else:
            self.model.router = None

    def setup_budget_functions(self):
        """Setup budget calculation functions based on budget type."""
        self._setup_param_loss_func()

        if self.budget_type == 'mem':
            self._setup_memory_loss_func()

    def setup_hooks(self):
        """Setup elasticity hooks on the model."""
        if getattr(self.config, 'flextron', False):
            self.hook_managers = apply_flextron_elasticity_to_model(self.model, self.config)

    def _setup_param_loss_func(self):
        """Setup parameter counting for budget calculations."""

        self.all_param, self.active_param = torch.tensor(
            get_num_parameters(
                hybrid_pattern=self.model.hybrid_layer_pattern,
                mamba_num_heads=self.config.mamba_num_heads,
                mamba_d_head=self.config.mamba_head_dim,
                mamba_d_state=self.config.mamba_state_dim,
                num_attention_heads=self.config.num_attention_heads,
                num_query_groups=self.config.num_query_groups,
                ffn_hidden_size=self.config.ffn_hidden_size,
                hidden_size=self.config.hidden_size,
                kv_channels=self.config.kv_channels,
                vocab_size=self.model.vocab_size,
                tied_vocab=self.model.share_embeddings_and_output_weights,
                num_experts=self.config.num_moe_experts,
                shared_expert_intermediate_size=self.config.moe_shared_expert_intermediate_size,
                moe_router_topk=self.config.moe_router_topk,
            ),
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )

    def _setup_memory_loss_func(self):
        """Setup memory loss function by calculating the baseline memory footprint."""
        self.total_memory = (
            get_memory_footprint(
                hybrid_pattern=self.model.hybrid_layer_pattern,
                mamba_num_heads=self.config.mamba_num_heads,
                mamba_d_head=self.config.mamba_head_dim,
                mamba_d_state=self.config.mamba_state_dim,
                num_attention_heads=self.config.num_attention_heads,
                num_query_groups=self.config.num_query_groups,
                ffn_hidden_size=self.config.ffn_hidden_size,
                hidden_size=self.config.hidden_size,
                kv_channels=self.config.kv_channels,
                vocab_size=self.model.vocab_size,
                tied_vocab=self.model.share_embeddings_and_output_weights,
                mem_infer_seq_len=self.config.mem_infer_seq_len,
                mem_batch_size=self.config.mem_batch_size,
                prefill_chunk_size=self.config.prefill_chunk_size,
                moe_num_experts=self.config.num_moe_experts,
                shared_expert_intermediate_size=self.config.moe_shared_expert_intermediate_size,
                moe_router_topk=self.config.moe_router_topk,
                memory_config=self.memory_config,
            )
            .float()
            .to(torch.cuda.current_device())
        )

        print(
            f"Total baseline memory footprint: {self.total_memory.item():.4f} GB  "
            f"(profile={getattr(get_args(), 'memory_profile', 'bf16')}, "
            f"param_target={self.memory_config.param_budget_target})"
        )

    def budget_loss_func(self, flextron_kwargs, budget_item=0, original_model=True):
        """Calculate budget-based loss exactly as in the original implementation."""
        dtype, device = (
            flextron_kwargs['router_mlp'][0].dtype,
            flextron_kwargs['router_mlp'][0].device,
        )

        flex_mamba_num_head = flextron_kwargs['router_mamba'][0] @ torch.tensor(
            self.config.mamba_int_list, dtype=dtype, device=device
        )
        flex_hidden_size = flextron_kwargs['router_emb'][0] @ torch.tensor(
            self.config.emb_int_list, dtype=dtype, device=device
        )
        flex_ffn_hidden_size = flextron_kwargs['router_mlp'][0] @ torch.tensor(
            self.config.mlp_int_list, dtype=dtype, device=device
        )
        flex_moe_expert = flextron_kwargs['router_moe_expert'][0] @ torch.tensor(
            self.config.moe_expert_int_list, dtype=dtype, device=device
        )
        # Attention heads are not router-controlled; pass the parent value through.
        num_attention_heads = self.config.num_attention_heads

        if self.config.add_skipping:
            logit_skip_selected = torch.cumsum(flextron_kwargs['router_skip'][0], 0)[:-1]
            logit_skip_all = torch.ones(self.config.num_layers).to(dtype=dtype, device=device)
            logit_skip_all[self.config.layer_ranking_list] = logit_skip_selected

            mamba_idxs = [
                i for i, char in enumerate(self.model.hybrid_layer_pattern) if char == 'M'
            ]
            mamba_idxs = torch.tensor(mamba_idxs, dtype=torch.long)
            flex_mamba_num_head = flex_mamba_num_head * logit_skip_all[mamba_idxs]
            flex_mamba_num_head = flex_mamba_num_head.unsqueeze(-1)

            head_idxs = [i for i, char in enumerate(self.model.hybrid_layer_pattern) if char == '*']
            head_idxs = torch.tensor(head_idxs, dtype=torch.long)
            num_attention_heads = num_attention_heads * logit_skip_all[head_idxs]
            num_attention_heads = num_attention_heads.unsqueeze(-1)

            moe_idxs = [i for i, char in enumerate(self.model.hybrid_layer_pattern) if char == 'E']
            moe_idxs = torch.tensor(moe_idxs, dtype=torch.long)
            flex_ffn_hidden_size = flex_ffn_hidden_size * logit_skip_all[moe_idxs]
            flex_ffn_hidden_size = flex_ffn_hidden_size.unsqueeze(-1)

            flex_moe_expert = flex_moe_expert * logit_skip_all[moe_idxs]
            flex_moe_expert = flex_moe_expert.unsqueeze(-1)

        if not self.config.flex_hetero_ffn and not self.config.add_skipping:
            flex_ffn_hidden_size = flex_ffn_hidden_size.unsqueeze(-1)
        if not self.config.flex_hetero_mamba and not self.config.add_skipping:
            flex_mamba_num_head = flex_mamba_num_head.unsqueeze(-1)
        if not self.config.flex_hetero_moe_expert and not self.config.add_skipping:
            flex_moe_expert = flex_moe_expert.unsqueeze(-1)

        current_param_all, current_param_active = get_num_parameters(
            hybrid_pattern=self.model.hybrid_layer_pattern,
            mamba_num_heads=flex_mamba_num_head.float(),
            mamba_d_head=self.config.mamba_head_dim,
            mamba_d_state=self.config.mamba_state_dim,
            num_attention_heads=(
                num_attention_heads.float()
                if isinstance(num_attention_heads, torch.Tensor)
                else num_attention_heads
            ),
            num_query_groups=self.config.num_query_groups,
            ffn_hidden_size=flex_ffn_hidden_size.float(),
            hidden_size=flex_hidden_size.unsqueeze(-1).float(),
            kv_channels=self.config.kv_channels,
            vocab_size=self.model.vocab_size,
            tied_vocab=self.model.share_embeddings_and_output_weights,
            num_experts=flex_moe_expert.float(),
            shared_expert_intermediate_size=self.config.moe_shared_expert_intermediate_size,
            moe_router_topk=self.config.moe_router_topk,
        )


        if self.config.budget_type == 'param':
            if self.memory_config.param_budget_target == 'active':
                diff = abs(current_param_active / (budget_item * self.active_param) - 1)
            else:
                diff = abs(current_param_all / (budget_item * self.all_param) - 1)
        elif self.config.budget_type == 'mem':
            current_mem = get_memory_footprint(
                hybrid_pattern=self.model.hybrid_layer_pattern,
                mamba_num_heads=flex_mamba_num_head.float(),
                mamba_d_head=self.config.mamba_head_dim,
                mamba_d_state=self.config.mamba_state_dim,
                num_attention_heads=(
                    num_attention_heads.float()
                    if isinstance(num_attention_heads, torch.Tensor)
                    else num_attention_heads
                ),
                num_query_groups=self.config.num_query_groups,
                ffn_hidden_size=flex_ffn_hidden_size.float(),
                hidden_size=flex_hidden_size.unsqueeze(-1).float(),
                kv_channels=self.config.kv_channels,
                vocab_size=self.model.vocab_size,
                tied_vocab=self.model.share_embeddings_and_output_weights,
                mem_infer_seq_len=self.config.mem_infer_seq_len,
                mem_batch_size=self.config.mem_batch_size,
                prefill_chunk_size=self.config.prefill_chunk_size,
                moe_num_experts=flex_moe_expert.float(),
                shared_expert_intermediate_size=self.config.moe_shared_expert_intermediate_size,
                moe_router_topk=self.config.moe_router_topk,
                memory_config=self.memory_config,
            ).float()
            diff = abs(current_mem / budget_item - 1)
        else:
            raise ValueError(f"Invalid budget type: {self.config.budget_type}")

        # return current_param, {}
        if budget_item != 1.0 and diff < 0.05:
            diff = diff * 0.0

        # if getattr(self.config, 'disable_budget', False):
        #     diff = diff * 0.0

        if budget_item == 1.0:
            if self.config.flex_hetero_moe_expert:
                label_moe_expert = torch.zeros_like(flextron_kwargs['router_moe_expert'][0])
                label_moe_expert[:, 0] = 1.0
                mse_loss_moe_expert = F.mse_loss(
                    flextron_kwargs['router_moe_expert'][0], label_moe_expert
                )
            else:
                label_moe_expert = torch.zeros_like(flextron_kwargs['router_moe_expert'][0])
                label_moe_expert[0] = 1.0
                mse_loss_moe_expert = F.mse_loss(
                    flextron_kwargs['router_moe_expert'][0], label_moe_expert
                )

            if self.config.flex_hetero_mamba:
                label_mamba = torch.zeros_like(flextron_kwargs['router_mamba'][0])
                label_mamba[:, 0] = 1.0
                mse_loss_mamba = F.mse_loss(flextron_kwargs['router_mamba'][0], label_mamba)
            else:
                label_mamba = torch.zeros_like(flextron_kwargs['router_mamba'][0])
                label_mamba[0] = 1.0
                mse_loss_mamba = F.mse_loss(flextron_kwargs['router_mamba'][0], label_mamba)

            if self.config.flex_hetero_ffn:
                label_mlp = torch.zeros_like(flextron_kwargs['router_mlp'][0])
                label_mlp[:, 0] = 1.0
                mse_loss_mlp = F.mse_loss(flextron_kwargs['router_mlp'][0], label_mlp)
            else:
                label_mlp = torch.zeros_like(flextron_kwargs['router_mlp'][0])
                label_mlp[0] = 1.0
                mse_loss_mlp = F.mse_loss(flextron_kwargs['router_mlp'][0], label_mlp)

            if self.config.add_skipping:
                label_skip = torch.zeros_like(flextron_kwargs['router_skip'][0])
                label_skip[0] = 1.0
                mse_loss_skip = F.mse_loss(flextron_kwargs['router_skip'][0], label_skip)
            else:
                mse_loss_skip = 0.0

            label_emb = torch.zeros_like(flextron_kwargs['router_emb'][0])
            label_emb[0] = 1.0
            mse_loss_emb = F.mse_loss(flextron_kwargs['router_emb'][0], label_emb)

            diff += 10 * (
                mse_loss_mamba
                + mse_loss_mlp
                + mse_loss_moe_expert
                + mse_loss_skip
                + mse_loss_emb
            )

        if original_model:
            return diff * 0.0, {}
        else:
            return diff.bfloat16(), {}

    def get_loss_func(self):
        """Get the budget loss function."""
        return self.budget_loss_func

    def process_router_output(self, budget_item):
        """Process router output and return flextron_kwargs."""
        if self.router is None:
            return {}, None

        (router_mlp, router_skip, router_emb, router_mamba, router_moe_expert) = (
            self.router(budget_item)
        )

        flextron_kwargs = {
            'router_mlp': router_mlp,
            'router_skip': router_skip,
            'router_emb': router_emb,
            'router_mamba': router_mamba,
            'router_moe_expert': router_moe_expert,
        }

        return flextron_kwargs, self.get_loss_func()

    def update_hook_elasticity_params(self, flextron_kwargs):
        """Update elasticity parameters in all hook managers."""
        if not self.hook_managers:
            return

        # Extract elasticity parameters from router outputs
        router_emb = flextron_kwargs.get('router_emb')
        router_mamba = flextron_kwargs.get('router_mamba')
        router_mlp = flextron_kwargs.get('router_mlp')
        router_moe_expert = flextron_kwargs.get('router_moe_expert')
        router_skip = flextron_kwargs.get('router_skip')  # General layer skipping

        # Update all hook managers with router outputs directly
        for manager in self.hook_managers:
            if hasattr(manager, 'set_elasticity_params'):
                manager.set_elasticity_params(
                    router_emb=router_emb,
                    router_mamba=router_mamba,
                    router_mlp=router_mlp,
                    router_moe_expert=router_moe_expert,
                    router_skip=router_skip,
                )


def setup_flextron_model(model):
    """
    Setup Flextron functionality for a model after creation.

    Args:
        model: The HybridModel instance

    Returns:
        FlextronModelManager: Manager object to handle Flextron operations
    """
    manager = FlextronModelManager(model, model.config)

    # Setup all Flextron components
    manager.setup_router()
    manager.setup_budget_functions()
    manager.setup_hooks()

    # Store manager on model for easy access
    model._flextron_manager = manager

    return manager


def inject_flextron_forward_logic(model):
    """
    Inject Flextron-specific forward pass logic into the model.
    This replaces the router logic that was previously in HybridModel.forward().
    """
    original_forward = model.forward

    def flextron_forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        decoder_input=None,
        labels=None,
        inference_context=None,
        runtime_gather_output=None,
        *,
        inference_params=None,
        **flextron_kwargs,
    ):

        # Handle override budget settings
        if getattr(self.config, 'override_selected_budget', None) is not None:
            assert (
                self.config.is_flex_eval
            ), "Override selected budget should only be set in flex eval mode"
            if self.config.override_selected_budget[0] == 1.0:
                flextron_kwargs = {}
            else:
                flextron_kwargs = {'budget': self.config.override_selected_budget[0]}

        # Initialize budget_loss
        budget_loss = None

        # Handle router logic if enabled and model has Flextron manager
        if (
            hasattr(self, '_flextron_manager')
            and self._flextron_manager is not None
            and self._flextron_manager.router is not None
        ):

            if 'budget' in flextron_kwargs:
                budget_item = flextron_kwargs['budget']
                original_model = False
            else:
                budget_item = 1.0
                original_model = True

            # Get router output and loss function
            flextron_kwargs, loss_func = self._flextron_manager.process_router_output(budget_item)

            # Calculate loss
            if loss_func:
                budget_loss = loss_func(flextron_kwargs, budget_item, original_model)

            if original_model:
                flextron_kwargs = {}

            # Update hook elasticity parameters
            self._flextron_manager.update_hook_elasticity_params(flextron_kwargs)
        else:
            # If no Flextron manager, clear flextron_kwargs to avoid passing unknown args
            flextron_kwargs = {}

        # Call original forward with processed flextron_kwargs
        result = original_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_context,
            runtime_gather_output=runtime_gather_output,
            inference_params=inference_params,
        )

        # Handle return values based on training mode and flextron settings
        if labels is not None:
            loss = result if not isinstance(result, tuple) else result[0]

            if (
                hasattr(self, '_flextron_manager')
                and self._flextron_manager is not None
                and self._flextron_manager.router is not None
                and getattr(self.config, 'flextron', False)
                and not getattr(self.config, 'is_flex_eval', False)
            ):
                if mpu.is_pipeline_last_stage():
                    return loss, budget_loss
                else:
                    return loss
            else:
                # Evaluation mode or non-flextron, return loss only
                return loss
        else:
            # No labels, return logits
            return result

    # Replace the forward method
    model.forward = flextron_forward.__get__(model, model.__class__)
