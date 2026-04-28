# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Flextron Elasticity Hooks

Applies elasticity masking through PyTorch hooks without modifying original
modules. One manager class per module type (MambaMixer, SelfAttention,
TransformerLayer, MoELayer, TopKRouter, TEGroupedMLP, HybridStack).
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.tensor_parallel.utils import split_tensor_along_last_dim


class FlextronMambaElasticityManager:
    """
    Manages elasticity for MambaMixer using pure PyTorch hooks.
    Based on the exact implementation from original flextron_os MambaMixer.
    """

    def __init__(self, config, layer_idx=0):
        self.config = config
        self.layer_idx = layer_idx
        self.enabled = getattr(config, 'flextron', False)

        if not self.enabled:
            return

        # Current elasticity parameters - store the full router outputs
        self.current_router_emb = None
        self.current_router_mamba = None

        # Hook handles for cleanup
        self.hook_handles = []

    def _init_embedding_masks(self):
        """Initialize embedding dimension masks."""
        mask_list = []
        for emb_int in self.config.emb_int_list:
            assert (
                0 <= emb_int <= self.config.hidden_size
            ), f'emb_int_list entries must be in [0, hidden_size={self.config.hidden_size}], got {emb_int}.'
            mask = torch.zeros(self.config.hidden_size, dtype=torch.bool)
            mask[:emb_int] = True
            mask_list.append(mask)
        self.emb_masks_lookup = {
            emb_int: idx for idx, emb_int in enumerate(self.config.emb_int_list)
        }
        self.emb_masks = torch.stack(mask_list, dim=0).to(device='cuda').to(dtype=torch.bfloat16)

    def _init_mamba_masks(self):
        """Initialize Mamba-specific masks for different layers."""
        in_proj_mask_list = []
        conv1d_mask_list = []
        out_proj_mask_list = []

        world_size = parallel_state.get_tensor_model_parallel_world_size()

        in_proj_z_shard = [i for i in range(self.mamba_mixer.d_inner_local_tp)]
        in_proj_x_shard = [
            i
            for i in range(self.mamba_mixer.d_inner_local_tp, 2 * self.mamba_mixer.d_inner_local_tp)
        ]
        in_proj_B_shard = [
            i
            for i in range(
                2 * self.mamba_mixer.d_inner_local_tp,
                2 * self.mamba_mixer.d_inner_local_tp
                + self.mamba_mixer.ngroups_local_tp * self.mamba_mixer.d_state,
            )
        ]
        in_proj_C_shard = [
            i
            for i in range(
                2 * self.mamba_mixer.d_inner_local_tp
                + self.mamba_mixer.ngroups_local_tp * self.mamba_mixer.d_state,
                2 * self.mamba_mixer.d_inner_local_tp
                + 2 * self.mamba_mixer.ngroups_local_tp * self.mamba_mixer.d_state,
            )
        ]
        in_proj_dt_shard = [
            i
            for i in range(
                2 * self.mamba_mixer.d_inner_local_tp
                + 2 * self.mamba_mixer.ngroups_local_tp * self.mamba_mixer.d_state,
                2 * self.mamba_mixer.d_inner_local_tp
                + 2 * self.mamba_mixer.ngroups_local_tp * self.mamba_mixer.d_state
                + self.mamba_mixer.nheads_local_tp,
            )
        ]

        conv1d_x_shard = [i for i in range(self.mamba_mixer.d_inner_local_tp)]
        conv1d_B_shard = [
            i
            for i in range(
                self.mamba_mixer.d_inner_local_tp,
                self.mamba_mixer.d_inner_local_tp
                + self.mamba_mixer.ngroups_local_tp * self.mamba_mixer.d_state,
            )
        ]
        conv1d_C_shard = [
            i
            for i in range(
                self.mamba_mixer.d_inner_local_tp
                + self.mamba_mixer.ngroups_local_tp * self.mamba_mixer.d_state,
                self.mamba_mixer.d_inner_local_tp
                + 2 * self.mamba_mixer.ngroups_local_tp * self.mamba_mixer.d_state,
            )
        ]

        out_proj_x_shard = [i for i in range(self.mamba_mixer.d_inner_local_tp)]

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        for mamba_int in self.config.mamba_int_list:
            assert (
                0 <= mamba_int <= self.mamba_mixer.nheads
            ), "mamba_int_list entries must be in [0, nheads={self.mamba_mixer.nheads}], got {mamba_int}."
            assert (
                mamba_int % tp_size == 0
            ), "mamba_int_list entries must be evenly divisible by tp_size={tp_size}, got {mamba_int}."
            mamba_nhead_idx = mamba_int // tp_size

            in_proj_mask = torch.zeros(
                self.mamba_mixer.d_inner_local_tp * 2
                + self.mamba_mixer.ngroups_local_tp * self.mamba_mixer.d_state * 2
                + self.mamba_mixer.nheads_local_tp,
                dtype=torch.bool,
            )
            in_proj_mask[in_proj_z_shard[: mamba_nhead_idx * self.mamba_mixer.headdim]] = True
            in_proj_mask[in_proj_x_shard[: mamba_nhead_idx * self.mamba_mixer.headdim]] = True
            in_proj_mask[in_proj_B_shard] = True
            in_proj_mask[in_proj_C_shard] = True
            in_proj_mask[in_proj_dt_shard[:mamba_nhead_idx]] = True
            in_proj_mask_list.append(in_proj_mask)

            conv1d_mask = torch.zeros(
                self.mamba_mixer.d_inner_local_tp
                + self.mamba_mixer.ngroups_local_tp * self.mamba_mixer.d_state * 2,
                dtype=torch.bool,
            )
            conv1d_mask[conv1d_x_shard[: mamba_nhead_idx * self.mamba_mixer.headdim]] = True
            conv1d_mask[conv1d_B_shard] = True
            conv1d_mask[conv1d_C_shard] = True
            conv1d_mask_list.append(conv1d_mask)

            out_proj_mask = torch.zeros(self.mamba_mixer.d_inner_local_tp, dtype=torch.bool)
            out_proj_mask[out_proj_x_shard[: mamba_nhead_idx * self.mamba_mixer.headdim]] = True
            out_proj_mask_list.append(out_proj_mask)
        self.mamba_masks_lookup = {
            mamba_int: idx for idx, mamba_int in enumerate(self.config.mamba_int_list)
        }

        in_proj_mask_list = [item.to(in_proj_mask_list[0].device) for item in in_proj_mask_list]
        self.in_proj_mask_list = (
            torch.stack(in_proj_mask_list, dim=0).to(device='cuda').to(dtype=torch.bfloat16)
        )

        conv1d_mask_list = [item.to(conv1d_mask_list[0].device) for item in conv1d_mask_list]
        self.conv1d_mask_list = (
            torch.stack(conv1d_mask_list, dim=0).to(device='cuda').to(dtype=torch.bfloat16)
        )

    def attach_hooks(self, mamba_mixer):
        """Attach hooks to MambaMixer following the original flextron_os pattern."""
        if not self.enabled:
            return

        self.mamba_mixer = mamba_mixer

        emb_effective_per_list = [x / self.config.hidden_size for x in self.config.emb_int_list]
        mamba_effective_per_list = [x / mamba_mixer.nheads for x in self.config.mamba_int_list]

        # Setup hook - runs first to initialize masks for this forward pass
        def setup_masks_hook(module, input):
            if self.config.flextron:
                self._init_embedding_masks()
                self._init_mamba_masks()
            return input

        # Cleanup hook - runs last to remove masks after forward pass
        def cleanup_masks_hook(module, input, output):
            if self.config.flextron:
                self.emb_masks = None
                self.in_proj_mask_list = None
                self.conv1d_mask_list = None
                self.out_proj_mask_list = None
                self.emb_masks_lookup = {}
                self.mamba_masks_lookup = {}
            return output

        # Hook 1: Input masking and router_emb processing
        def input_mask_hook(module, input):
            if self.config.flextron and self.current_router_emb is not None:
                hidden_states = input[0]

                # Apply embedding mask
                if self.config.soft_mask:
                    soft_mask = torch.zeros(
                        self.emb_masks[0].shape,
                        dtype=torch.bfloat16,
                        device=self.emb_masks[0].device,
                    )
                    for mask, per_logit in zip(self.emb_masks, self.current_router_emb[0]):
                        soft_mask.add_(mask * per_logit)
                    mask = soft_mask
                    masked_input = hidden_states * mask[None, None, :]
                else:
                    router_emb_logits, emb_choice = (
                        torch.max(self.current_router_emb[0]),
                        self.current_router_emb[1],
                    )
                    mask = self.emb_masks[self.emb_masks_lookup[emb_choice]]
                    masked_input = hidden_states * mask[None, None, :]
                    masked_input = masked_input * router_emb_logits

                return tuple([masked_input] + list(input[1:]))

            return input

        # Hook 2: in_proj pre-hook for eps modification
        def in_proj_pre_hook(module, input):
            if self.config.flextron and self.current_router_emb is not None:

                # Set eps to the pruned value
                if self.config.soft_mask:
                    soft_eps = 0
                    for emb_per, per_logit in zip(
                        emb_effective_per_list, self.current_router_emb[0]
                    ):
                        soft_eps += self.config.layernorm_epsilon * emb_per * per_logit
                    module.eps = soft_eps.float().detach().item()
                else:
                    emb_choice = self.current_router_emb[1]
                    emb_effective_per = emb_choice / self.config.hidden_size
                    module.eps = self.config.layernorm_epsilon * emb_effective_per
            return input

        # Hook 3: in_proj post-hook for router scaling
        def in_proj_post_hook(module, input, output):
            if self.config.flextron and self.current_router_mamba is not None:
                # Apply router_emb scaling to in_proj output
                xz, bias = output

                if self.config.soft_mask:
                    # Soft scaling with embedding router
                    soft_xz = torch.zeros_like(xz)
                    for emb_per, per_logit in zip(
                        emb_effective_per_list, self.current_router_emb[0]
                    ):
                        soft_xz.add_(xz * per_logit * (emb_per**0.5))
                    xz = soft_xz
                else:
                    router_emb_logits, emb_choice = (
                        torch.max(self.current_router_emb[0]),
                        self.current_router_emb[1],
                    )
                    emb_effective_per = emb_choice / self.config.hidden_size
                    xz = xz * router_emb_logits * (emb_effective_per**0.5)

                # Apply mamba router logic (hard mask only)
                if not self.config.soft_mask:
                    if self.config.flex_hetero_mamba:
                        mamba_idx = (
                            self.config.hybrid_layer_pattern[: self.layer_idx + 1].count('M') - 1
                        )
                        router_mamba_logits = torch.max(self.current_router_mamba[0][mamba_idx])
                        mamba_per = self.current_router_mamba[1][mamba_idx]
                    else:
                        router_mamba_logits, mamba_per = (
                            torch.max(self.current_router_mamba[0]),
                            self.current_router_mamba[1],
                        )

                # Apply mamba masking
                if self.config.soft_mask:
                    soft_in_proj_mask = torch.zeros_like(self.in_proj_mask_list[0])
                    if self.config.flex_hetero_mamba:
                        mamba_idx = (
                            self.config.hybrid_layer_pattern[: self.layer_idx + 1].count('M') - 1
                        )
                        for mask, per_logit in zip(
                            self.in_proj_mask_list, self.current_router_mamba[0][mamba_idx]
                        ):
                            soft_in_proj_mask.add_(mask * per_logit)
                    else:
                        for mask, per_logit in zip(
                            self.in_proj_mask_list, self.current_router_mamba[0]
                        ):
                            soft_in_proj_mask.add_(mask * per_logit)
                    in_proj_mask = soft_in_proj_mask
                else:
                    in_proj_mask = self.in_proj_mask_list[self.mamba_masks_lookup[mamba_per]]

                xz = xz * in_proj_mask.to(device=xz.device)[None, None, :]

                if not self.config.soft_mask:
                    xz = xz * router_mamba_logits

                # Reset eps to original
                module.eps = self.config.layernorm_epsilon

                return (xz, bias)
            return output

        # Hook 4: conv1d output masking
        def conv1d_mask_hook(module, input, output):
            if self.config.flextron and self.current_router_mamba is not None:
                if not self.config.soft_mask:
                    if self.config.flex_hetero_mamba:
                        mamba_idx = (
                            self.config.hybrid_layer_pattern[: self.layer_idx + 1].count('M') - 1
                        )
                        router_mamba_logits = torch.max(self.current_router_mamba[0][mamba_idx])
                        mamba_per = self.current_router_mamba[1][mamba_idx]
                    else:
                        router_mamba_logits, mamba_per = (
                            torch.max(self.current_router_mamba[0]),
                            self.current_router_mamba[1],
                        )

                # Apply conv1d masking
                if self.config.soft_mask:
                    soft_conv1d_mask = torch.zeros_like(self.conv1d_mask_list[0])
                    if self.config.flex_hetero_mamba:
                        mamba_idx = (
                            self.config.hybrid_layer_pattern[: self.layer_idx + 1].count('M') - 1
                        )
                        for mask, per_logit in zip(
                            self.conv1d_mask_list, self.current_router_mamba[0][mamba_idx]
                        ):
                            soft_conv1d_mask.add_(mask * per_logit)
                    else:
                        for mask, per_logit in zip(
                            self.conv1d_mask_list, self.current_router_mamba[0]
                        ):
                            soft_conv1d_mask.add_(mask * per_logit)
                    conv1d_mask = soft_conv1d_mask
                else:
                    conv1d_mask = self.conv1d_mask_list[self.mamba_masks_lookup[mamba_per]]
                masked_output = output * conv1d_mask.to(device=output.device)[None, :, None]

                if not self.config.soft_mask:
                    masked_output = masked_output * router_mamba_logits

                return masked_output
            return output

        # Hook 5a: RMSNorm pre-hook for eps modification
        def norm_pre_hook(module, input):
            if self.config.flextron and self.current_router_mamba is not None:
                if self.config.soft_mask:
                    soft_eps = 0
                    if self.config.flex_hetero_mamba:
                        mamba_idx = (
                            self.config.hybrid_layer_pattern[: self.layer_idx + 1].count('M') - 1
                        )
                        for mamba_per, per_logit in zip(
                            mamba_effective_per_list, self.current_router_mamba[0][mamba_idx]
                        ):
                            soft_eps += self.config.layernorm_epsilon * mamba_per * per_logit
                    else:
                        for mamba_per, per_logit in zip(
                            mamba_effective_per_list, self.current_router_mamba[0]
                        ):
                            soft_eps += self.config.layernorm_epsilon * mamba_per * per_logit
                    module.eps = soft_eps.float().detach().item()
                else:
                    if self.config.flex_hetero_mamba:
                        mamba_idx = (
                            self.config.hybrid_layer_pattern[: self.layer_idx + 1].count('M') - 1
                        )
                        mamba_per = self.current_router_mamba[1][mamba_idx]
                    else:
                        mamba_per = self.current_router_mamba[1]
                    mamba_effective_per = mamba_per / self.mamba_mixer.nheads
                    module.eps = self.config.layernorm_epsilon * mamba_effective_per

            return input

        # Hook 5b: RMSNorm post-hook for scaling and eps restoration
        def norm_post_hook(module, input, output):
            if self.config.flextron and self.current_router_mamba is not None:
                # Restore original eps
                module.eps = self.config.layernorm_epsilon

                if self.config.soft_mask:
                    soft_scaled_output = torch.zeros_like(output)
                    if self.config.flex_hetero_mamba:
                        mamba_idx = (
                            self.config.hybrid_layer_pattern[: self.layer_idx + 1].count('M') - 1
                        )
                        for mamba_per, per_logit in zip(
                            mamba_effective_per_list, self.current_router_mamba[0][mamba_idx]
                        ):
                            soft_scaled_output.add_(output * (mamba_per**0.5) * per_logit)
                    else:
                        for mamba_per, per_logit in zip(
                            mamba_effective_per_list, self.current_router_mamba[0]
                        ):
                            soft_scaled_output.add_(output * (mamba_per**0.5) * per_logit)
                    return soft_scaled_output
                else:
                    if self.config.flex_hetero_mamba:
                        mamba_idx = (
                            self.config.hybrid_layer_pattern[: self.layer_idx + 1].count('M') - 1
                        )
                        router_mamba_logits = torch.max(self.current_router_mamba[0][mamba_idx])
                        mamba_per = self.current_router_mamba[1][mamba_idx]
                    else:
                        router_mamba_logits, mamba_per = (
                            torch.max(self.current_router_mamba[0]),
                            self.current_router_mamba[1],
                        )
                    mamba_effective_per = mamba_per / self.mamba_mixer.nheads
                    return output * (mamba_effective_per**0.5) * router_mamba_logits

            return output

        # Hook 6: Final output masking
        def output_mask_hook(module, input, output):
            if self.config.flextron and self.current_router_emb is not None:
                out, out_bias = output

                # Apply embedding mask
                if self.config.soft_mask:
                    soft_mask = torch.zeros(
                        self.emb_masks[0].shape,
                        dtype=torch.bfloat16,
                        device=self.emb_masks[0].device,
                    )
                    for mask, per_logit in zip(self.emb_masks, self.current_router_emb[0]):
                        soft_mask.add_(mask * per_logit)
                    mask = soft_mask
                    masked_out = out * mask[None, None, :]
                else:
                    router_emb_logits, emb_choice = (
                        torch.max(self.current_router_emb[0]),
                        self.current_router_emb[1],
                    )
                    mask = self.emb_masks[self.emb_masks_lookup[emb_choice]]
                    masked_out = out * mask[None, None, :]
                    masked_out = masked_out * router_emb_logits

                return (masked_out, out_bias)
            return output

        # IMPORTANT: Register setup hook FIRST
        setup_handle = mamba_mixer.register_forward_pre_hook(setup_masks_hook)
        self.hook_handles.append(setup_handle)

        # Attach main input hook
        main_handle = mamba_mixer.register_forward_pre_hook(input_mask_hook)
        self.hook_handles.append(main_handle)

        # Attach in_proj hooks
        in_proj_pre_handle = mamba_mixer.in_proj.register_forward_pre_hook(in_proj_pre_hook)
        in_proj_post_handle = mamba_mixer.in_proj.register_forward_hook(in_proj_post_hook)
        self.hook_handles.append(in_proj_pre_handle)
        self.hook_handles.append(in_proj_post_handle)

        # Attach conv1d hook (this will handle the standard conv1d path)
        conv_handle = mamba_mixer.conv1d.register_forward_hook(conv1d_mask_hook)
        self.hook_handles.append(conv_handle)

        # Attach RMSNorm hooks if rmsnorm is enabled
        norm_pre_handle = mamba_mixer.norm.register_forward_pre_hook(norm_pre_hook)
        norm_post_handle = mamba_mixer.norm.register_forward_hook(norm_post_hook)
        self.hook_handles.append(norm_pre_handle)
        self.hook_handles.append(norm_post_handle)

        # Final output hook
        output_handle = mamba_mixer.register_forward_hook(output_mask_hook)
        self.hook_handles.append(output_handle)

        # Cleanup hook - runs last to remove masks after forward pass
        cleanup_handle = mamba_mixer.register_forward_hook(cleanup_masks_hook)
        self.hook_handles.append(cleanup_handle)

    def set_elasticity_params(self, router_emb=None, router_mamba=None, **kwargs):
        """Set current elasticity parameters that will be used by hooks."""
        if router_emb is not None:
            self.current_router_emb = router_emb

        if router_mamba is not None:
            self.current_router_mamba = router_mamba

    def detach_hooks(self):
        """Remove all hooks."""
        if not hasattr(self, 'hook_handles'):
            return
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def __del__(self):
        """Cleanup hooks when manager is destroyed."""
        self.detach_hooks()


class FlextronTransformerLayerElasticityManager:
    """
    Manages elasticity for TransformerLayer using pure PyTorch hooks.
    Handles input/pre-MLP layernorm eps modification and MLP routing.
    """

    def __init__(self, config, layer_idx=0):
        self.config = config
        self.layer_idx = layer_idx
        self.enabled = getattr(config, 'flextron', False)

        if not self.enabled:
            return

        # Current elasticity parameters - store the full router outputs
        self.current_router_emb = None

        # Hook handles for cleanup
        self.hook_handles = []

    def _init_embedding_masks(self):
        """Initialize embedding dimension masks."""
        mask_list = []
        for emb_int in self.config.emb_int_list:
            mask = torch.zeros(self.config.hidden_size, dtype=torch.bool)
            mask[:emb_int] = True
            mask_list.append(mask)
        self.emb_masks_lookup = {
            emb_int: idx for idx, emb_int in enumerate(self.config.emb_int_list)
        }
        self.emb_masks = torch.stack(mask_list, dim=0).to(device='cuda').to(dtype=torch.bfloat16)

    def initialize_masks(self, transformer_layer):
        """Initialize masks based on the MoE module configuration."""
        if not self.enabled:
            return

        self.transformer_layer = transformer_layer
        self._init_embedding_masks()

    def attach_hooks(self, transformer_layer):
        """Attach hooks to MLP/MoE layer for layer skipping only."""
        if not self.enabled:
            return

        self.initialize_masks(transformer_layer)

        emb_effective_per_list = [x / self.config.hidden_size for x in self.config.emb_int_list]

        # Hook 2: Pre-MLP layernorm pre-hook for eps modification
        def pre_mlp_layernorm_pre_hook(module, input):

            if self.config.flextron and self.current_router_emb is not None:
                hidden_states = input[0]
                # Apply embedding mask
                if self.config.soft_mask:
                    soft_mask = torch.zeros(
                        self.emb_masks[0].shape,
                        dtype=torch.bfloat16,
                        device=self.emb_masks[0].device,
                    )
                    for mask, per_logit in zip(self.emb_masks, self.current_router_emb[0]):
                        soft_mask.add_(mask * per_logit)
                    mask = soft_mask
                    masked_input = hidden_states * mask[None, None, :]
                else:
                    router_emb_logits, emb_choice = (
                        torch.max(self.current_router_emb[0]),
                        self.current_router_emb[1],
                    )
                    mask = self.emb_masks[self.emb_masks_lookup[emb_choice]]
                    masked_input = hidden_states * mask[None, None, :]
                    masked_input = masked_input * router_emb_logits

                # Modify eps for this forward pass
                if self.config.soft_mask:
                    soft_eps = 0
                    for emb_per, per_logit in zip(
                        emb_effective_per_list, self.current_router_emb[0]
                    ):
                        soft_eps += self.config.layernorm_epsilon * emb_per * per_logit
                    module.eps = soft_eps.float().detach().item()
                else:
                    emb_choice = self.current_router_emb[1]
                    emb_effective_per = emb_choice / self.config.hidden_size
                    module.eps = self.config.layernorm_epsilon * emb_effective_per

                return tuple([masked_input] + list(input[1:]))

            return input

        # Hook 3: Pre-MLP layernorm post-hook for scaling and eps restoration
        def pre_mlp_layernorm_post_hook(module, input, output):
            if self.config.flextron and self.current_router_emb is not None:

                # Restore original eps
                module.eps = self.config.layernorm_epsilon

                # Apply scaling
                if self.config.soft_mask:
                    soft_scaled_output = torch.zeros_like(output)
                    for emb_per, per_logit in zip(
                        emb_effective_per_list, self.current_router_emb[0]
                    ):
                        soft_scaled_output.add_(output * (emb_per**0.5) * per_logit)
                    scaled_output = soft_scaled_output
                else:
                    emb_choice = self.current_emb_choice
                    emb_effective_per = emb_choice / self.config.hidden_size
                    router_emb_logits = torch.max(self.current_router_emb[0])
                    scaled_output = output * (emb_effective_per**0.5) * router_emb_logits
                return scaled_output

            return output

        # Attach the pre-MLP layernorm hooks
        pre_mlp_ln_pre_handle = transformer_layer.pre_mlp_layernorm.register_forward_pre_hook(
            pre_mlp_layernorm_pre_hook
        )
        pre_mlp_ln_post_handle = transformer_layer.pre_mlp_layernorm.register_forward_hook(
            pre_mlp_layernorm_post_hook
        )
        self.hook_handles.append(pre_mlp_ln_pre_handle)
        self.hook_handles.append(pre_mlp_ln_post_handle)

    def set_elasticity_params(self, router_emb=None, **kwargs):
        """Set current elasticity parameters that will be used by hooks."""
        if router_emb is not None:
            self.current_router_emb = router_emb
            self.current_emb_choice = router_emb[1]

    def detach_hooks(self):
        """Remove all hooks."""
        if not hasattr(self, 'hook_handles'):
            return
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def __del__(self):
        """Cleanup hooks when manager is destroyed."""
        self.detach_hooks()


def topk_softmax_with_capacity(
    logits: torch.Tensor,
    topk: int,
    capacity_factor: Optional[float] = None,
    pad_to_capacity: bool = False,
    drop_policy: str = "probs",
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    deterministic_mode: bool = False,
    score_function: str = "softmax",
    expert_bias: Optional[torch.Tensor] = None,
    current_router_moe_expert_0: Optional[torch.Tensor] = None,
    current_router_moe_expert_per: Optional[List[float]] = None,
    num_experts: int = 0,
):
    """Apply capacity and padding to the top-k selection.
    Args:
        logits (torch.Tensor): Logits tensor.
        topk (int): The number of experts to select for each token.
        capacity_factor (float): The capacity factor of each expert. Will drop tokens if the number
                               of tokens exceeds the capacity.
        pad_to_capacity (bool): Whether to need padding in token drop mode. The probs for padded
                               tokens will be 0.
        drop_policy (str): The policy to drop tokens. Can be either "prob" or "position".
                           If "prob", the tokens with the lowest probabilities will be dropped.
                           If "position", tokens at the end of each batch will be dropped.
        use_pre_softmax (bool): Whether to apply softmax or sigmoid before top-k selection.
        num_groups (int): Number of groups for routed experts.
        group_topk (int): Number of selected groups for each token.
        scaling_factor (float): Scaling factor of routing score in top-k selection.
        deterministic_mode (bool): Deprecated.
        score_function (str): The score function to use. Can be either "softmax" or "sigmoid".
        expert_bias (torch.Tensor): The bias added to logits for expert routing.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - routing_probs (torch.Tensor): A tensor of shape [num_tokens, num_experts] containing
              the routing probabilities for each token to each expert.
            - routing_map (torch.Tensor): A mask tensor of shape [num_tokens, num_experts]
              indicating which experts were selected for each token. True values represent
              the selected experts.
            - tokens_per_expert (torch.Tensor): A tensor of shape [num_experts] containing
              the number of local tokens assigned to each expert before dropping and padding.
    """
    assert score_function == "sigmoid", "Only sigmoid score function is supported for now."
    assert expert_bias is not None, "Expert bias is required for sigmoid score function."
    assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
    num_tokens, num_experts = logits.shape

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if group_topk:
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        else:
            return torch.topk(scores, k=topk, dim=1)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
    elif score_function == "sigmoid":
        # logits[:, 96:] = float('-inf')
        # expert_bias[96:] = 0
        scores_for_routing = 0
        scores_for_topk = 0
        for router_moe_expert_logits, router_moe_expert_per in zip(
            current_router_moe_expert_0, current_router_moe_expert_per
        ):
            expert_threshold = math.floor(router_moe_expert_per * num_experts)

            logits_current = logits.clone()
            logits_current[:, expert_threshold:] = float('-inf')
            expert_bias_current = expert_bias.clone()
            expert_bias_current[expert_threshold:] = 0
            expert_bias_current = expert_bias_current * router_moe_expert_logits

            scores = (
                torch.sigmoid(logits_current.float()).type_as(logits_current)
                * router_moe_expert_logits
            )

            scores_for_topk += scores
            scores_for_routing += scores + expert_bias_current

        _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
        scores = torch.gather(scores_for_topk, dim=1, index=top_indices).type_as(logits)

        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores

    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    # TODO Try using element-wise operations instead of scatter?
    topk_masked_gates = torch.zeros_like(logits).scatter(1, top_indices, probs)
    topk_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    tokens_per_expert = topk_map.sum(dim=0)

    if capacity_factor is None:
        # TopK without capacity
        return topk_masked_gates, topk_map, tokens_per_expert
    else:
        # TopK with capacity
        expert_capacity = get_capacity(
            num_tokens=num_tokens * topk, num_experts=num_experts, capacity_factor=capacity_factor
        )

        # Maskout exceeded tokens
        if drop_policy == "probs":
            _, capacity_indices = torch.topk(
                topk_masked_gates, k=expert_capacity, dim=0, sorted=False
            )
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1).bool()
        elif drop_policy == "position":
            _, capacity_indices = torch.topk(topk_map.int(), k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1).bool()
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

        if pad_to_capacity:
            final_map = capacity_mask
            final_probs = topk_masked_gates * final_map
        else:
            # Get exceed mask and maskout exceeded probs and indices
            final_map = torch.logical_and(topk_map, capacity_mask)
            final_probs = topk_masked_gates * final_map
        return final_probs, final_map, tokens_per_expert


class FlextronTopKRouterElasticityManager:
    """
    Manages elasticity for MoE Router using pure PyTorch hooks.
    Handles expert masking in the routing logits before topk selection.
    """

    def __init__(self, config, layer_idx=0):
        self.config = config
        self.layer_idx = layer_idx
        self.enabled = getattr(config, 'flextron', False)

        if not self.enabled:
            return

        # Current elasticity parameters
        self.current_router_moe_expert = None

        # Hook handles for cleanup
        self.hook_handles = []

    def attach_hooks(self, router):
        """Attach hooks to TopKRouter for expert masking."""
        if not self.enabled:
            return

        # Store original method for restoration
        original_routing = router.routing

        def wrapped_routing(logits, **kwargs):

            # Apply expert masking before calling original routing
            if self.config.flextron and self.current_router_moe_expert is not None:

                if self.config.soft_mask:
                    if self.config.flex_hetero_moe_expert:
                        moe_expert_idx = (
                            self.config.hybrid_layer_pattern[: self.layer_idx + 1].count('E') - 1
                        )
                        current_router_moe_expert_0 = self.current_router_moe_expert[0][
                            moe_expert_idx
                        ]
                    else:
                        current_router_moe_expert_0 = self.current_router_moe_expert[0]

                    seq_length, bsz = logits.shape[:2]
                    logits = logits.view(-1, self.config.num_moe_experts)

                    # Apply Z-Loss
                    logits = router.apply_z_loss(logits)
                    assert self.config.moe_router_load_balancing_type == "none"

                    scores, routing_map, _ = topk_softmax_with_capacity(
                        logits,
                        self.config.moe_router_topk,
                        capacity_factor=self.config.moe_expert_capacity_factor,
                        pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                        drop_policy=self.config.moe_token_drop_policy,
                        use_pre_softmax=self.config.moe_router_pre_softmax,
                        num_groups=self.config.moe_router_num_groups,
                        group_topk=self.config.moe_router_group_topk,
                        scaling_factor=self.config.moe_router_topk_scaling_factor,
                        deterministic_mode=self.config.deterministic_mode,
                        score_function=self.config.moe_router_score_function,
                        expert_bias=router.expert_bias,
                        current_router_moe_expert_0=current_router_moe_expert_0,
                        current_router_moe_expert_per=[
                            x / self.config.num_moe_experts for x in self.config.moe_expert_int_list
                        ],
                        num_experts=self.config.num_moe_experts,
                    )

                    if self.config.moe_router_enable_expert_bias and torch.is_grad_enabled():
                        with torch.no_grad():
                            router.local_tokens_per_expert += routing_map.sum(dim=0)

                    return scores, routing_map
                else:
                    if self.config.flex_hetero_moe_expert:
                        moe_expert_idx = (
                            self.config.hybrid_layer_pattern[: self.layer_idx + 1].count('E') - 1
                        )
                        router_moe_expert_logits = torch.max(
                            self.current_router_moe_expert[0][moe_expert_idx]
                        )
                        router_moe_expert_per = self.current_router_moe_expert[1][moe_expert_idx]
                    else:
                        router_moe_expert_logits, router_moe_expert_per = (
                            torch.max(self.current_router_moe_expert[0]),
                            self.current_router_moe_expert[1],
                        )

                    expert_threshold = (
                        router_moe_expert_per  # always an integer count after conversion
                    )

                    # Apply the same logic as the commented lines
                    logits = logits.clone()
                    logits[:, expert_threshold:] = float('-inf')
                    logits = logits * router_moe_expert_logits

                    # Also handle expert bias
                    if hasattr(router, 'expert_bias') and router.expert_bias is not None:
                        router.expert_bias = router.expert_bias.clone()
                        router.expert_bias[expert_threshold:] = 0

                    return original_routing(logits, **kwargs)

            else:
                return original_routing(logits, **kwargs)

        router.routing = wrapped_routing
        # Store reference to restore later
        router._original_routing = original_routing
        self.hook_handles.append(('method_replacement', router, 'routing'))

    def set_elasticity_params(self, router_moe_expert=None, **kwargs):
        """Set current elasticity parameters that will be used by hooks."""
        if router_moe_expert is not None:
            self.current_router_moe_expert = router_moe_expert

    def detach_hooks(self):
        """Remove all hooks and restore original methods."""
        if not hasattr(self, 'hook_handles'):
            return
        for handle in self.hook_handles:
            if isinstance(handle, tuple) and handle[0] == 'method_replacement':
                # Restore original method
                _, router, method_name = handle
                if hasattr(router, '_original_routing'):
                    router.routing = router._original_routing
                    delattr(router, '_original_routing')
            else:
                # Regular hook handle
                handle.remove()
        self.hook_handles.clear()

    def __del__(self):
        """Cleanup hooks when manager is destroyed."""
        self.detach_hooks()


class FlextronMoEElasticityManager:
    """
    Manages elasticity for MLP/MoE layers using pure PyTorch hooks.
    Now supports both traditional MLP ('-') and MoE ('E') layers with layer skipping.
    """

    def __init__(self, config, layer_idx=0):
        self.config = config
        self.layer_idx = layer_idx
        self.enabled = getattr(config, 'flextron', False)

        if not self.enabled:
            return

        # Current elasticity parameters - store the full router outputs
        self.current_router_emb = None
        # Hook handles for cleanup
        self.hook_handles = []

    def _init_embedding_masks(self):
        """Initialize embedding dimension masks."""
        mask_list = []
        for emb_int in self.config.emb_int_list:
            mask = torch.zeros(self.config.hidden_size, dtype=torch.bool)
            mask[:emb_int] = True
            mask_list.append(mask)
        self.emb_masks_lookup = {
            emb_int: idx for idx, emb_int in enumerate(self.config.emb_int_list)
        }
        self.emb_masks = torch.stack(mask_list, dim=0).to(device='cuda').to(dtype=torch.bfloat16)

    def initialize_masks(self, moe_module):
        """Initialize masks based on the MoE module configuration."""
        if not self.enabled:
            return

        self.moe_module = moe_module
        self._init_embedding_masks()

    def attach_hooks(self, moe_module):
        """Attach hooks to MLP/MoE layer for layer skipping only."""
        if not self.enabled:
            return

        self.initialize_masks(moe_module)

        def output_mask_hook(module, input, output):

            if self.config.flextron and self.current_router_emb is not None:
                out, out_bias = output

                if self.config.soft_mask:
                    soft_mask = torch.zeros(
                        self.emb_masks[0].shape,
                        dtype=torch.bfloat16,
                        device=self.emb_masks[0].device,
                    )
                    for mask, per_logit in zip(self.emb_masks, self.current_router_emb[0]):
                        soft_mask.add_(mask * per_logit)
                    mask = soft_mask
                    masked_out = out * mask[None, None, :]
                else:
                    router_emb_logits, emb_choice = (
                        torch.max(self.current_router_emb[0]),
                        self.current_router_emb[1],
                    )
                    mask = self.emb_masks[self.emb_masks_lookup[emb_choice]]
                    masked_out = out * mask[None, None, :]
                    masked_out = masked_out * router_emb_logits

                return (masked_out, out_bias)
            return output

        # Attach the output hook

        output_handle = moe_module.register_forward_hook(output_mask_hook)
        self.hook_handles.append(output_handle)

    def set_elasticity_params(self, router_emb=None, **kwargs):
        """Set current elasticity parameters that will be used by hooks."""
        if router_emb is not None:
            self.current_router_emb = router_emb

    def detach_hooks(self):
        """Remove all hooks."""
        if not hasattr(self, 'hook_handles'):
            return
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def __del__(self):
        """Cleanup hooks when manager is destroyed."""
        self.detach_hooks()


class FlextronGroupedMLPElasticityManager:

    def __init__(self, config, layer_idx=0):
        self.config = config
        self.layer_idx = layer_idx
        self.enabled = getattr(config, 'flextron', False)
        self.mlp_idx = self.config.hybrid_layer_pattern[: self.layer_idx + 1].count('E') - 1

        if not self.enabled:
            return

        self.current_router_mlp = None
        self.current_router_emb = None

        self.hook_handles = []

    def _init_embedding_masks(self):
        """Initialize embedding dimension masks."""
        mask_list = []
        for emb_int in self.config.emb_int_list:
            mask = torch.zeros(self.config.hidden_size, dtype=torch.bool)
            mask[:emb_int] = True
            mask_list.append(mask)
        self.emb_masks_lookup = {
            emb_int: idx for idx, emb_int in enumerate(self.config.emb_int_list)
        }
        self.emb_masks = torch.stack(mask_list, dim=0).to(device='cuda').to(dtype=torch.bfloat16)

    def _init_mlp_masks(self):
        """Initialize MLP-specific masks."""
        mask_list = []
        list_mlp_mask = list(set(self.config.mlp_int_list))
        list_mlp_mask.sort(reverse=True)
        for mlp_int in list_mlp_mask:
            mask_temp = torch.zeros(self.config.ffn_hidden_size, dtype=torch.bool)
            mask_temp[:mlp_int] = True
            mask_list.append(mask_temp)
        mask_list = [item.to(mask_list[0].device) for item in mask_list]
        self.mlp_intermediate_masks = (
            torch.stack(mask_list, dim=0).to(device='cuda').to(dtype=torch.bfloat16)
        )
        self.mlp_intermediate_masks_lookup = {
            mlp_int: idx for idx, mlp_int in enumerate(list_mlp_mask)
        }

    def initialize_masks(self, mlp_module):
        """Initialize masks based on the MLP configuration."""
        if not self.enabled:
            return

        self.mlp_module = mlp_module
        self._init_embedding_masks()
        self._init_mlp_masks()

    def attach_hooks(self, mlp_module):
        """Attach hooks to MLP following the original flextron_os pattern."""
        if not self.enabled:
            return

        self.mlp_module = mlp_module

        emb_effective_per_list = [x / self.config.hidden_size for x in self.config.emb_int_list]

        # Setup hook - runs first to initialize masks for this forward pass
        def setup_masks_hook(module, input):
            if self.config.flextron:
                self._init_embedding_masks()
                self._init_mlp_masks()
            return input

        # Cleanup hook - runs last to remove masks after forward pass
        def cleanup_masks_hook(module, input, output):
            if self.config.flextron:
                self.emb_masks = None
                self.mlp_intermediate_masks = None
                self.emb_masks_lookup = {}
                self.mlp_intermediate_masks_lookup = {}
            return output

        # IMPORTANT: Register setup hook FIRST
        setup_handle = mlp_module.register_forward_pre_hook(setup_masks_hook)
        self.hook_handles.append(setup_handle)

        # Hook 1: Input masking and router_emb processing
        def input_mask_hook(module, input):
            if self.config.flextron and self.current_router_emb is not None:
                hidden_states = input[0]

                # Apply embedding mask
                if self.config.soft_mask:
                    soft_mask = torch.zeros(
                        self.emb_masks[0].shape,
                        dtype=torch.bfloat16,
                        device=self.emb_masks[0].device,
                    )
                    for mask, per_logit in zip(self.emb_masks, self.current_router_emb[0]):
                        soft_mask.add_(mask * per_logit)
                    mask = soft_mask
                    masked_input = hidden_states * mask[None, :]
                else:
                    router_emb_logits, emb_choice = (
                        torch.max(self.current_router_emb[0]),
                        self.current_router_emb[1],
                    )
                    mask = self.emb_masks[self.emb_masks_lookup[emb_choice]]
                    masked_input = hidden_states * mask[None, :]
                    masked_input = masked_input * router_emb_logits

                # Process router_mlp logic here
                router_weights = None
                if self.current_router_mlp is not None:
                    if self.config.flex_hetero_ffn:
                        router_weights = torch.max(self.current_router_mlp[0][self.mlp_idx])
                        mlp_per = self.current_router_mlp[1][self.mlp_idx]
                    else:
                        router_weights, mlp_per = (
                            torch.max(self.current_router_mlp[0]),
                            self.current_router_mlp[1],
                        )

                # Store router info for later hooks
                module._flextron_router_weights = router_weights
                module._flextron_mlp_per = mlp_per

                return tuple([masked_input] + list(input[1:]))
            return input

        # Hook 2: Linear FC1 post-hook for router scaling and masking
        def fc1_post_hook(module, input, output):

            # Apply router_emb scaling
            if self.config.flextron and self.current_router_mlp is not None:
                intermediate_parallel, bias_parallel = output
                if self.config.soft_mask:
                    soft_intermediate_parallel = torch.zeros_like(intermediate_parallel)
                    for emb_per, per_logit in zip(
                        emb_effective_per_list, self.current_router_emb[0]
                    ):
                        soft_intermediate_parallel.add_(intermediate_parallel * per_logit)
                    intermediate_parallel = soft_intermediate_parallel
                else:
                    router_emb_logits, emb_choice = (
                        torch.max(self.current_router_emb[0]),
                        self.current_router_emb[1],
                    )
                    intermediate_parallel = intermediate_parallel * router_emb_logits

                # # Apply MLP masking and router weights

                mlp_per = mlp_module._flextron_mlp_per
                router_weights = getattr(mlp_module, '_flextron_router_weights', None)

                # Apply masking
                if self.config.soft_mask:
                    soft_mask = torch.zeros(
                        self.mlp_intermediate_masks[0].shape,
                        dtype=torch.bfloat16,
                        device=self.mlp_intermediate_masks[0].device,
                    )
                    if self.config.flex_hetero_ffn:
                        for mask, per_logit in zip(
                            self.mlp_intermediate_masks, self.current_router_mlp[0][self.mlp_idx]
                        ):
                            soft_mask.add_(mask * per_logit)
                    else:
                        for mask, per_logit in zip(
                            self.mlp_intermediate_masks, self.current_router_mlp[0]
                        ):
                            soft_mask.add_(mask * per_logit)
                    mask = soft_mask
                else:
                    mask = self.mlp_intermediate_masks[self.mlp_intermediate_masks_lookup[mlp_per]]

                world_size = parallel_state.get_expert_tensor_parallel_world_size()
                mask_list = split_tensor_along_last_dim(mask, world_size)
                rank = parallel_state.get_expert_tensor_parallel_rank()

                mask = mask_list[rank].contiguous()

                intermediate_parallel = (
                    intermediate_parallel * mask.to(device=intermediate_parallel.device)[None, :]
                )
                if router_weights is not None and not self.config.soft_mask:
                    intermediate_parallel = intermediate_parallel * router_weights

                module.eps = self.config.layernorm_epsilon

                return (intermediate_parallel, bias_parallel)
            return output

        # Hook 3: Final output masking
        def output_mask_hook(module, input, output):
            if self.config.flextron and self.current_router_emb is not None:
                out, out_bias = output

                # Apply embedding mask
                if self.config.soft_mask:
                    soft_mask = torch.zeros(
                        self.emb_masks[0].shape,
                        dtype=torch.bfloat16,
                        device=self.emb_masks[0].device,
                    )
                    for mask, per_logit in zip(self.emb_masks, self.current_router_emb[0]):
                        soft_mask.add_(mask * per_logit)
                    mask = soft_mask
                    masked_out = out * mask[None, :]
                else:
                    router_emb_logits, emb_choice = (
                        torch.max(self.current_router_emb[0]),
                        self.current_router_emb[1],
                    )
                    mask = self.emb_masks[self.emb_masks_lookup[emb_choice]]
                    masked_out = out * mask[None, :]
                    masked_out = masked_out * router_emb_logits

                return (masked_out, out_bias)
            return output

        # Hook 1: Input masking and router_emb processing
        main_handle = mlp_module.register_forward_pre_hook(input_mask_hook)
        self.hook_handles.append(main_handle)

        # Hook 2: Linear FC1 pre-hook for eps modification
        fc1_post_handle = mlp_module.linear_fc1.register_forward_hook(fc1_post_hook)
        self.hook_handles.append(fc1_post_handle)

        # Hook 3: Final output masking
        output_handle = mlp_module.register_forward_hook(output_mask_hook)
        self.hook_handles.append(output_handle)

        # Cleanup hook - runs last to remove masks after forward pass
        cleanup_handle = mlp_module.register_forward_hook(cleanup_masks_hook)
        self.hook_handles.append(cleanup_handle)

    def set_elasticity_params(self, router_emb=None, router_mlp=None, **kwargs):
        """Set current elasticity parameters that will be used by hooks."""
        if router_emb is not None:
            self.current_router_emb = router_emb

        if router_mlp is not None:
            self.current_router_mlp = router_mlp

    def detach_hooks(self):
        """Remove all hooks."""
        if not hasattr(self, 'hook_handles'):
            return
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def __del__(self):
        """Cleanup hooks when manager is destroyed."""
        self.detach_hooks()


class FlextronAttentionElasticityManager:
    """
    Manages elasticity for Attention using pure PyTorch hooks.
    Based on the exact implementation from original flextron_os Attention.
    """

    def __init__(self, config, layer_idx=0):
        self.config = config
        self.layer_idx = layer_idx
        self.enabled = getattr(config, 'flextron', False)

        if not self.enabled:
            return

        # Current elasticity parameters - store the full router outputs
        self.current_router_emb = None
        self.current_router_head = None

        # Hook handles for cleanup
        self.hook_handles = []

    def _init_embedding_masks(self):
        """Initialize embedding dimension masks."""
        mask_list = []
        for emb_int in self.config.emb_int_list:
            mask = torch.zeros(self.config.hidden_size, dtype=torch.bool)
            mask[:emb_int] = True
            mask_list.append(mask)
        self.emb_masks_lookup = {
            emb_int: idx for idx, emb_int in enumerate(self.config.emb_int_list)
        }
        self.emb_masks = torch.stack(mask_list, dim=0).to(device='cuda').to(dtype=torch.bfloat16)

    def _init_head_masks(self):
        """Initialize attention head masks."""
        mask_list = []
        full_dim = self.attention_module.num_attention_heads_per_partition * self.config.kv_channels
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        for head_int in self.config.head_int_list:
            # head_int is total heads; convert to per-partition dimension
            num_heads = (head_int // tp_size) * self.config.kv_channels
            mask_temp = torch.zeros(full_dim, dtype=torch.bool)
            mask_temp[:num_heads] = True
            mask_list.append(mask_temp)
        self.head_masks_lookup = {
            head_int: idx for idx, head_int in enumerate(self.config.head_int_list)
        }

        self.head_masks = torch.stack(mask_list, dim=0).to(device='cuda').to(dtype=torch.bfloat16)

    def attach_hooks(self, attention_module):
        """Attach hooks to Attention following the original flextron_os pattern."""
        if not self.enabled:
            return

        self.attention_module = attention_module

        emb_effective_per_list = [x / self.config.hidden_size for x in self.config.emb_int_list]

        # Setup hook - runs first to initialize masks for this forward pass
        def setup_masks_hook(module, input):
            if self.config.flextron:
                self._init_embedding_masks()
                self._init_head_masks()
            return input

        # Cleanup hook - runs last to remove masks after forward pass
        def cleanup_masks_hook(module, input, output):
            if self.config.flextron:
                self.emb_masks = None
                self.head_masks = None
                self.emb_masks_lookup = {}
                self.head_masks_lookup = {}
            return output

        # IMPORTANT: Register setup hook FIRST
        setup_handle = attention_module.register_forward_pre_hook(setup_masks_hook)
        self.hook_handles.append(setup_handle)

        # Hook 1: Input masking and router_emb processing
        def input_mask_hook(module, input):
            if self.config.flextron and self.current_router_emb is not None:
                hidden_states = input[0]

                # Apply embedding mask
                if self.config.soft_mask:
                    soft_mask = torch.zeros(
                        self.emb_masks[0].shape,
                        dtype=torch.bfloat16,
                        device=self.emb_masks[0].device,
                    )
                    for mask, per_logit in zip(self.emb_masks, self.current_router_emb[0]):
                        soft_mask.add_(mask * per_logit)
                    mask = soft_mask
                    masked_input = hidden_states * mask[None, None, :]
                else:
                    router_emb_logits, emb_choice = (
                        torch.max(self.current_router_emb[0]),
                        self.current_router_emb[1],
                    )
                    mask = self.emb_masks[self.emb_masks_lookup[emb_choice]]
                    masked_input = (
                        hidden_states * mask.to(device=hidden_states.device)[None, None, :]
                    )
                    masked_input = masked_input * router_emb_logits

                return tuple([masked_input] + list(input[1:]))
            return input

        # Hook 2: Linear QKV pre-hook for eps modification
        def linear_qkv_pre_hook(module, input):
            if self.config.flextron and self.current_router_emb is not None:
                # Set eps on linear_qkv (fused layernorm)
                if self.config.soft_mask:
                    soft_eps = 0
                    for emb_per, per_logit in zip(
                        emb_effective_per_list, self.current_router_emb[0]
                    ):
                        soft_eps += self.config.layernorm_epsilon * emb_per * per_logit
                    module.eps = soft_eps.float().detach().item()
                else:
                    router_emb_logits, emb_choice = (
                        torch.max(self.current_router_emb[0]),
                        self.current_router_emb[1],
                    )
                    emb_effective_per = emb_choice / self.config.hidden_size
                    module.eps = self.config.layernorm_epsilon * emb_effective_per

            return input

        # Hook 3: Linear QKV post-hook for scaling
        def linear_qkv_post_hook(module, input, output):
            if self.config.flextron and self.current_router_emb is not None:
                query_key_value, bias = output
                if self.config.soft_mask:
                    soft_query_key_value = torch.zeros_like(query_key_value)
                    for emb_per, per_logit in zip(
                        emb_effective_per_list, self.current_router_emb[0]
                    ):
                        soft_query_key_value.add_(query_key_value * (emb_per**0.5) * per_logit)
                    scaled_output = soft_query_key_value
                else:
                    router_emb_logits, emb_choice = (
                        torch.max(self.current_router_emb[0]),
                        self.current_router_emb[1],
                    )
                    emb_effective_per = emb_choice / self.config.hidden_size
                    scaled_output = query_key_value * router_emb_logits * (emb_effective_per**0.5)
                module.eps = self.config.layernorm_epsilon
                return (scaled_output, bias)
            return output

        # Hook 4: Core attention output masking
        def core_attention_mask_hook(module, input, output):
            if self.current_router_head is not None:
                # Apply head masking
                if self.config.soft_mask:
                    soft_mask = torch.zeros(
                        self.head_masks[0].shape,
                        dtype=torch.bfloat16,
                        device=self.head_masks[0].device,
                    )
                    if self.config.flex_hetero_head:
                        head_idx = (
                            self.config.hybrid_layer_pattern[: self.layer_idx + 1].count('*') - 1
                        )
                        for mask, per_logit in zip(
                            self.head_masks, self.current_router_head[0][head_idx]
                        ):
                            soft_mask.add_(mask * per_logit)
                    else:
                        for mask, per_logit in zip(self.head_masks, self.current_router_head[0]):
                            soft_mask.add_(mask * per_logit)
                    mask = soft_mask
                    masked_output = output * mask[None, None, :]
                else:
                    # Hard masking fallback
                    if hasattr(attention_module, '_flextron_head_per'):
                        head_per = attention_module._flextron_head_per
                        router_head_weights = getattr(
                            attention_module, '_flextron_router_head_weights', None
                        )
                        mask = self.head_masks[self.head_masks_lookup[head_per]]
                        masked_output = output * mask[None, None, :]
                        if router_head_weights is not None:
                            masked_output = masked_output * router_head_weights
                    else:
                        masked_output = output
                        mask = None

                return masked_output
            return output

        # Hook 5: Final output masking
        def output_mask_hook(module, input, output):
            if self.config.flextron and self.current_router_emb is not None:
                out, out_bias = output

                # Apply embedding mask
                if self.config.soft_mask:
                    soft_mask = torch.zeros(
                        self.emb_masks[0].shape,
                        dtype=torch.bfloat16,
                        device=self.emb_masks[0].device,
                    )
                    for mask, per_logit in zip(self.emb_masks, self.current_router_emb[0]):
                        soft_mask.add_(mask * per_logit)
                    mask = soft_mask
                    masked_out = out * mask[None, None, :]
                else:
                    router_emb_logits, emb_choice = (
                        torch.max(self.current_router_emb[0]),
                        self.current_router_emb[1],
                    )
                    mask = self.emb_masks[self.emb_masks_lookup[emb_choice]]
                    masked_out = out * mask[None, None, :]
                    masked_out = masked_out * router_emb_logits

                return (masked_out, out_bias)
            return output

        # Hook 1: Input masking and router_emb processing
        main_handle = attention_module.register_forward_pre_hook(input_mask_hook)
        self.hook_handles.append(main_handle)

        # Hook 2&3: Linear QKV pre-hook for eps modification
        qkv_pre_handle = attention_module.linear_qkv.register_forward_pre_hook(linear_qkv_pre_hook)
        qkv_post_handle = attention_module.linear_qkv.register_forward_hook(linear_qkv_post_hook)
        self.hook_handles.append(qkv_pre_handle)
        self.hook_handles.append(qkv_post_handle)

        # Final output masking
        output_handle = attention_module.register_forward_hook(output_mask_hook)
        self.hook_handles.append(output_handle)

        # Cleanup hook - runs last to remove masks after forward pass
        cleanup_handle = attention_module.register_forward_hook(cleanup_masks_hook)
        self.hook_handles.append(cleanup_handle)

    def set_elasticity_params(self, router_emb=None, router_head=None, **kwargs):
        """Set current elasticity parameters that will be used by hooks."""
        if router_emb is not None:
            self.current_router_emb = router_emb

        if router_head is not None:
            self.current_router_head = router_head

    def detach_hooks(self):
        """Remove all hooks."""
        if not hasattr(self, 'hook_handles'):
            return
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def __del__(self):
        """Cleanup hooks when manager is destroyed."""
        self.detach_hooks()


class FlextronStackElasticityManager:
    """
    Manages elasticity for HybridStack using pure PyTorch hooks.
    Handles input masking and final norm scaling.
    """

    def __init__(self, config):
        self.config = config
        self.enabled = getattr(config, 'flextron', False)

        if not self.enabled:
            return

        # Current elasticity parameters
        self.current_emb_choice = self.config.hidden_size
        self.current_router_emb = None

        # Hook handles for cleanup
        self.hook_handles = []

        # Pre-computed masks
        self.emb_masks = None
        self.emb_masks_lookup = {}

    def initialize_masks(self, stack):
        """Initialize masks based on the stack configuration."""
        if not self.enabled:
            return

        self.stack = stack

    def attach_hooks(self, stack):
        """Attach hooks to HybridStack."""
        if not self.enabled:
            return

        self.initialize_masks(stack)

        emb_effective_per_list = [x / self.config.hidden_size for x in self.config.emb_int_list]

        # Hook 1: Final norm pre-hook for eps modification
        def final_norm_pre_hook(module, input):
            if self.config.flextron and self.current_router_emb is not None:
                # Modify eps for this forward pass
                if self.config.soft_mask:
                    soft_eps = 0
                    for emb_per, per_logit in zip(
                        emb_effective_per_list, self.current_router_emb[0]
                    ):
                        soft_eps += self.config.layernorm_epsilon * emb_per * per_logit
                    module.eps = soft_eps.float().detach().item()
                else:
                    emb_choice = self.current_emb_choice
                    emb_effective_per = emb_choice / self.config.hidden_size
                    module.eps = self.config.layernorm_epsilon * emb_effective_per

            return input

        # Hook 2: Final norm post-hook for scaling and eps restoration
        def final_norm_post_hook(module, input, output):
            if self.config.flextron and self.current_router_emb is not None:
                # Restore original eps
                module.eps = self.config.layernorm_epsilon

                # Apply scaling
                if self.config.soft_mask:
                    soft_scaled_output = torch.zeros_like(output)
                    for emb_per, per_logit in zip(
                        emb_effective_per_list, self.current_router_emb[0]
                    ):
                        soft_scaled_output.add_(output * (emb_per**0.5) * per_logit)
                    scaled_output = soft_scaled_output
                else:
                    emb_choice = self.current_emb_choice
                    emb_effective_per = emb_choice / self.config.hidden_size
                    router_emb_logits = torch.max(self.current_router_emb[0])
                    scaled_output = output * (emb_effective_per**0.5) * router_emb_logits
                return scaled_output

            return output

        # Hooks for final norm if it exists
        final_norm_pre_handle = stack.final_norm.register_forward_pre_hook(final_norm_pre_hook)
        final_norm_post_handle = stack.final_norm.register_forward_hook(final_norm_post_hook)
        self.hook_handles.append(final_norm_pre_handle)
        self.hook_handles.append(final_norm_post_handle)

    def set_elasticity_params(self, router_emb=None, **kwargs):
        """Set current elasticity parameters that will be used by hooks."""
        if router_emb is not None:
            self.current_router_emb = router_emb
            self.current_emb_choice = router_emb[1]

    def detach_hooks(self):
        """Remove all hooks and restore original forward method."""
        if not hasattr(self, 'hook_handles'):
            return
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def __del__(self):
        """Cleanup hooks when manager is destroyed."""
        self.detach_hooks()


def add_flextron_mamba_elasticity(mamba_mixer, config, layer_idx=0):
    """
    Add elasticity to a MambaMixer using hooks.

    Args:
        mamba_mixer: The MambaMixer instance to add elasticity to
        config: Configuration object with flextron settings
        layer_idx: Index of this layer in the hybrid pattern

    Returns:
        FlextronMambaElasticityManager: Manager object to control elasticity
    """
    if hasattr(mamba_mixer, '_flextron_manager'):
        return mamba_mixer._flextron_manager
    manager = FlextronMambaElasticityManager(config, layer_idx)
    manager.attach_hooks(mamba_mixer)

    # Store manager reference on the mixer for easy access
    mamba_mixer._flextron_manager = manager

    return manager


def add_flextron_transformer_layer_elasticity(transformer_layer, config, layer_idx=0):
    """
    Add elasticity to a TransformerLayer using hooks.

    Args:
        transformer_layer: The TransformerLayer instance to add elasticity to
        config: Configuration object with flextron settings
        layer_idx: Index of this layer in the hybrid pattern

    Returns:
        FlextronTransformerLayerElasticityManager: Manager object to control elasticity
    """
    if hasattr(transformer_layer, '_flextron_layer_manager'):
        return transformer_layer._flextron_layer_manager
    manager = FlextronTransformerLayerElasticityManager(config, layer_idx)
    manager.attach_hooks(transformer_layer)

    # Store manager reference on the layer for easy access
    transformer_layer._flextron_layer_manager = manager

    return manager


def add_flextron_topk_router_elasticity(router, config, layer_idx=0):
    """
    Add elasticity to a TopKRouter using hooks.

    Args:
        router: The TopKRouter instance to add elasticity to
        config: Configuration object with flextron settings
        layer_idx: Index of this layer in the hybrid pattern

    Returns:
        FlextronTopKRouterElasticityManager: Manager object to control elasticity
    """
    if hasattr(router, '_flextron_router_manager'):
        return router._flextron_router_manager
    manager = FlextronTopKRouterElasticityManager(config, layer_idx)
    manager.attach_hooks(router)

    # Store manager reference on the router for easy access
    router._flextron_router_manager = manager

    return manager


def add_flextron_moe_elasticity(moe_module, config, layer_idx=0):
    """
    Add elasticity to a MoE using hooks.

    Args:
        moe_module: The MoE instance to add elasticity to
        config: Configuration object with flextron settings
        layer_idx: Index of this layer in the hybrid pattern

    Returns:
        FlextronMoEElasticityManager: Manager object to control elasticity
    """
    if hasattr(moe_module, '_flextron_manager'):
        return moe_module._flextron_manager
    manager = FlextronMoEElasticityManager(config, layer_idx)
    manager.attach_hooks(moe_module)

    # Store manager reference on the module for easy access
    moe_module._flextron_manager = manager

    return manager


def add_flextron_grouped_mlp_elasticity(grouped_mlp_module, config, layer_idx=0):
    """
    Add elasticity to a GroupedMLP using hooks.
    """
    if hasattr(grouped_mlp_module, '_flextron_manager'):
        return grouped_mlp_module._flextron_manager
    manager = FlextronGroupedMLPElasticityManager(config, layer_idx)
    manager.attach_hooks(grouped_mlp_module)

    # Store manager reference on the module for easy access
    grouped_mlp_module._flextron_manager = manager

    return manager


def add_flextron_attention_elasticity(attention_module, config, layer_idx=0):
    """
    Add elasticity to an Attention module using hooks.

    Args:
        attention_module: The Attention instance to add elasticity to
        config: Configuration object with flextron settings
        layer_idx: Index of this layer in the hybrid pattern

    Returns:
        FlextronAttentionElasticityManager: Manager object to control elasticity
    """
    if hasattr(attention_module, '_flextron_manager'):
        return attention_module._flextron_manager
    manager = FlextronAttentionElasticityManager(config, layer_idx)
    manager.attach_hooks(attention_module)

    # Store manager reference on the module for easy access
    attention_module._flextron_manager = manager

    return manager


def add_flextron_stack_elasticity(stack, config):
    """
    Add elasticity to a HybridStack using hooks.

    Args:
        stack: The HybridStack instance to add elasticity to
        config: Configuration object with flextron settings

    Returns:
        FlextronStackElasticityManager: Manager object to control elasticity
    """
    if hasattr(stack, '_flextron_manager'):
        return stack._flextron_manager
    manager = FlextronStackElasticityManager(config)
    manager.attach_hooks(stack)

    # Store manager reference on the stack for easy access
    stack._flextron_manager = manager

    return manager


# Convenience function to apply elasticity to all modules in a model
def apply_flextron_elasticity_to_model(model, config):
    """Apply elasticity to all MambaMixer, MLP/MoE, and Attention instances in a model based on hybrid pattern."""
    managers = []

    if not hasattr(config, 'hybrid_layer_pattern') or not config.hybrid_layer_pattern:
        # No hybrid pattern, skip elasticity setup
        return managers

    hybrid_pattern = config.hybrid_layer_pattern

    # Find decoder layers
    decoder = getattr(model, 'decoder', None)
    layers = getattr(decoder, 'layers', None)

    if decoder is None or layers is None:
        return managers

    # Apply elasticity per layer based on hybrid pattern
    for layer_idx, layer_char in enumerate(hybrid_pattern):
        if layer_idx >= len(layers):
            break

        layer = layers[layer_idx]

        if layer_char == 'E':  # MoE layer (treated as MLP replacement)
            if (
                'MoETransformerLayer' == layer.__class__.__name__
                or 'TransformerLayer' == layer.__class__.__name__
            ):
                layer_manager = add_flextron_transformer_layer_elasticity(layer, config, layer_idx)
                managers.append(layer_manager)

            # Find MoELayer module in this layer
            moe_module = None
            for name, module in layer.named_modules():
                if 'MoELayer' == module.__class__.__name__:
                    moe_module = module
                    break
            if moe_module is not None:
                manager = add_flextron_moe_elasticity(moe_module, config, layer_idx)
                managers.append(manager)

                # Also add router elasticity to the MoE router
                router_module = None
                for name, module in moe_module.named_modules():
                    if 'TopKRouter' == module.__class__.__name__:
                        router_module = module
                        break
                if router_module is not None:
                    router_manager = add_flextron_topk_router_elasticity(
                        router_module, config, layer_idx
                    )
                    managers.append(router_manager)

            # Find TEGroupedMLP module in this layer
            moe_module = None
            for name, module in layer.named_modules():
                if 'TEGroupedMLP' == module.__class__.__name__:
                    moe_module = module
                    break
            if moe_module is not None:
                manager = add_flextron_grouped_mlp_elasticity(moe_module, config, layer_idx)
                managers.append(manager)

        elif layer_char == 'M':  # Mamba layer
            mamba_module = None
            for name, module in layer.named_modules():
                if 'MambaMixer' == module.__class__.__name__:
                    mamba_module = module
                    break
            if mamba_module is not None:
                manager = add_flextron_mamba_elasticity(mamba_module, config, layer_idx)
                managers.append(manager)

        elif layer_char == '*':  # Attention layer (TransformerLayer)
            attention_module = None
            for name, module in layer.named_modules():
                if 'SelfAttention' == module.__class__.__name__:
                    attention_module = module
                    break
            if attention_module is not None:
                manager = add_flextron_attention_elasticity(attention_module, config, layer_idx)
                managers.append(manager)

    # Also add hooks to HybridStack if present
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'final_norm'):
        stack_manager = add_flextron_stack_elasticity(model.decoder, config)
        managers.append(stack_manager)

    # Store all managers on the model
    model._flextron_managers = managers
    return managers
